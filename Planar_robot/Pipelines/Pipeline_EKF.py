"""
This file contains the class Pipeline_EKF, 
which is used to train and test KalmanNet.
"""

import torch
import torch.nn as nn
import random
import time
import matplotlib.pyplot as plt
import numpy as np

class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model
        
    def adjust_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr   
        self.learningRate = new_lr

    def setTrainingParams(self, args, unsupervised, minimize_with_ut):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch # Number of Samples in Batch
        self.learningRate = args.lr # Learning Rate
        self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
        self.alpha = args.alpha # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.unsupervised = unsupervised
        self.minimize_with_ut = minimize_with_ut
        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
    
    # torch.autograd.set_detect_anomaly(True)
    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, \
        MaskOnState=False, randomInit=False,cv_init=None,train_init=None,\
        train_lengthMask=None,cv_lengthMask=None,\
            to_save_model_parameters = False, load_parameters_from_pre_training = False, pre_trained_model_path= None,train_delta_theta_obs = None, cv_delta_theta_obs= None ,train_angles_true = None , cv_angles_true = None, combined_loss = False):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])
        
        self.MSE_cv_linear_epoch_obs = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch_obs = torch.zeros([self.N_steps])
        
        self.MSE_cv_linear_epoch_combined_loss = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch_combined_loss = torch.zeros([self.N_steps])
        
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        
        self.MSE_train_linear_epoch_combined_loss = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch_combined_loss = torch.zeros([self.N_steps])
        
        self.MSE_train_linear_epoch_obs = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch_obs = torch.zeros([self.N_steps])
        
        # Load pre-trained parameters if the flag is set
        if load_parameters_from_pre_training and pre_trained_model_path is not None:
            self.model.load_state_dict(torch.load(pre_trained_model_path))
            print('Pre-trained model parameters loaded.')
        
        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])
        
        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B
            # Init Hidden State
            self.model.init_hidden_KNet()

            # Init Training Batch tensors
            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            #train_target_batch.requires_grad_(True)
            x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            y_out_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            noises_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            y_training_batch_prev = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            train_delta_theta_obs_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            train_first_values = torch.zeros([self.N_B, SysModel.n]).to(self.device)
            #x_out_training_batch.requires_grad_(True)
            if self.args.randomLength:
                MSE_train_linear_LOSS = torch.zeros([self.N_B])
                MSE_cv_linear_LOSS = torch.zeros([self.N_CV])

            # Randomly select N_B training sequences
            assert self.N_B <= self.N_E # N_B must be smaller than N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            index_list = [0] * len(n_e)
            ii = 0
            for index in n_e:
                index_list[ii] = index
                if self.args.randomLength:
                    y_training_batch[ii,:,train_lengthMask[index,:]] = train_input[index,:,train_lengthMask[index,:]]
                    train_target_batch[ii,:,train_lengthMask[index,:]] = train_target[index,:,train_lengthMask[index,:]]
                else:
                    y_training_batch[ii,:,:] = train_input[index]
                    train_target_batch[ii,:,:] = train_target[index] 
                    if SysModel.to_use_noises:
                        noises_training_batch[ii,:,:] = SysModel.noises_train[index]
                ii += 1
            
            # Init Sequence
            if(randomInit):
                train_init_batch = torch.empty([self.N_B, SysModel.m,1]).to(self.device)
                ii = 0
                for index in n_e:
                    train_init_batch[ii,:,0] = torch.squeeze(train_init[index])
                    ii += 1
                self.model.InitSequence(train_init_batch, SysModel.T)
            else:
                if SysModel.is_m1x_0_true:
                    self.model.InitSequence(\
                    SysModel.m1x_0_train[index_list], SysModel.T)
                else:
                    self.model.InitSequence(\
                    SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_B,1,1), SysModel.T)
            
            # Forward Computation
            for t in range(0, SysModel.T):
                x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t],2)))
                y_out_training_batch[:, :, t] = torch.squeeze(self.model.m1y)
            # Compute Training Loss
            MSE_trainbatch_linear_LOSS = 0
            MSE_trainbatch_linear_LOSS_obs = 0
            if (self.args.CompositionLoss):
                y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
                for t in range(SysModel.T):
                    ################### added by Linoy ####### dim = 2 was missing
                    y_hat[:,:,t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_training_batch[:,:,t],dim=2)))

                if(MaskOnState):### FIXME: composition loss, y_hat may have different mask with x
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(x_out_training_batch[jj,mask,train_lengthMask[index]], train_target_batch[jj,mask,train_lengthMask[index]])+(1-self.alpha)*self.loss_fn(y_hat[jj,mask,train_lengthMask[index]], y_training_batch[jj,mask,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:                     
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])+(1-self.alpha)*self.loss_fn(y_hat[:,mask,:], y_training_batch[:,mask,:])
                else:# no mask on state
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(x_out_training_batch[jj,:,train_lengthMask[index]], train_target_batch[jj,:,train_lengthMask[index]])+(1-self.alpha)*self.loss_fn(y_hat[jj,:,train_lengthMask[index]], y_training_batch[jj,:,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:                
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(x_out_training_batch, train_target_batch)+(1-self.alpha)*self.loss_fn(y_hat, y_training_batch)
            
            else:# no composition loss
                if(MaskOnState):
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.loss_fn(x_out_training_batch[jj,mask,train_lengthMask[index]], train_target_batch[jj,mask,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch[:,mask,:], train_target_batch[:,mask,:])
                else: # no mask on state
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:# mask out the padded part when computing loss
                            MSE_train_linear_LOSS[jj] = self.loss_fn(x_out_training_batch[jj,:,train_lengthMask[index]], train_target_batch[jj,:,train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else: 
                        MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch[:,:2,:], train_target_batch[:,:2,:])
                        MSE_trainbatch_linear_LOSS_obs = self.loss_fn(y_out_training_batch,y_training_batch - x_out_training_batch[:,2:,:] ) if self.minimize_with_ut else self.loss_fn(y_out_training_batch,y_training_batch)
                        
            if combined_loss:
                final_loss = MSE_trainbatch_linear_LOSS_obs if self.unsupervised else (0.6*MSE_trainbatch_linear_LOSS_obs + MSE_trainbatch_linear_LOSS)
            else:
                final_loss = MSE_trainbatch_linear_LOSS_obs if self.unsupervised else MSE_trainbatch_linear_LOSS
            # dB Loss
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_linear_epoch_obs[ti] = MSE_trainbatch_linear_LOSS_obs.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])
            self.MSE_train_dB_epoch_obs[ti] = 10 * torch.log10(self.MSE_train_linear_epoch_obs[ti])
            self.MSE_train_linear_epoch_combined_loss[ti] = MSE_trainbatch_linear_LOSS_obs.item() + MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_dB_epoch_combined_loss[ti] = 10 * torch.log10(self.MSE_train_linear_epoch_combined_loss[ti])
            
            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            final_loss.backward(retain_graph=True)
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            self.model.batch_size = self.N_CV
            # Init Hidden State
            self.model.init_hidden_KNet()
            with torch.no_grad():

                SysModel.T_test = cv_input.size()[-1] # T_test is the maximum length of the CV sequences

                x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)
                y_out_cv_batch = torch.empty([self.N_CV, SysModel.n, SysModel.T_test]).to(self.device)
                y_cv_batch_prev = torch.empty([self.N_CV, SysModel.n, SysModel.T_test]).to(self.device)
                # Init Sequence
                if(randomInit):
                    if(cv_init==None):
                        self.model.InitSequence(\
                        SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_CV,1,1), SysModel.T_test)
                    else:
                        self.model.InitSequence(cv_init, SysModel.T_test)                       
                else:
                    if SysModel.is_m1x_0_validation_true:
                        self.model.InitSequence(SysModel.m1x_0_validation, SysModel.T_test)
                    else:    
                        self.model.InitSequence(\
                        SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_CV,1,1), SysModel.T_test)

                for t in range(0, SysModel.T_test):
                    x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input[:, :, t],2)))
                    y_out_cv_batch[:, :, t] = torch.squeeze(self.model.m1y)
                    
                # Compute CV Loss
                MSE_cvbatch_linear_LOSS = 0
                if(MaskOnState):
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,mask,cv_lengthMask[index]], cv_target[index,mask,cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:          
                        MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch[:,mask,:], cv_target[:,mask,:])
                else:
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(x_out_cv_batch[index,:,cv_lengthMask[index]], cv_target[index,:,cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:
                        MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch[:,:2,:], cv_target[:,:2,:])
                        MSE_cvbatch_linear_LOSS_obs = self.loss_fn(y_out_cv_batch , cv_input- x_out_cv_batch[:,2:,:]) if self.minimize_with_ut else self.loss_fn(y_out_cv_batch , cv_input) # when we have b_t
                        
                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_linear_epoch_obs[ti] = MSE_cvbatch_linear_LOSS_obs.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                self.MSE_cv_dB_epoch_obs[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch_obs[ti])
                self.MSE_cv_linear_epoch_combined_loss[ti] = MSE_cvbatch_linear_LOSS.item() + MSE_cvbatch_linear_LOSS_obs.item()
                self.MSE_cv_dB_epoch_combined_loss[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch_combined_loss[ti])
                if combined_loss:
                    relevant_loss = MSE_cvbatch_linear_LOSS_obs if self.unsupervised else  (0.6*MSE_cvbatch_linear_LOSS_obs + MSE_cvbatch_linear_LOSS)
                else:
                    relevant_loss = MSE_cvbatch_linear_LOSS_obs if self.unsupervised else MSE_cvbatch_linear_LOSS
                relevant_loss = 10 * torch.log10(relevant_loss)
            
                if (relevant_loss < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = relevant_loss
                    self.MSE_cv_idx_opt = ti
                    # save best model's parameters
                    torch.save(self.model, path_results + 'best-model.pt')
                    if to_save_model_parameters:
                        torch.save(self.model.state_dict(), pre_trained_model_path)
            
            if ti == self.args.epoch_thresh_for_lr and self.args.lr_change_flag:
                new_lr = self.learningRate* 10**(-1)                
                self.adjust_learning_rate(new_lr)
                
            ########################
            ### Training Summary ###
            ########################
            print(ti, "Hi MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")
                      
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")
        
        # TRain set learning curve
        plt.figure()
        plt.subplot(3,2,1)
        plt.plot(self.MSE_train_dB_epoch, label='Training Loss')
        plt.title('Training Learning Curve')
        plt.xlabel('Step')
        plt.ylabel('Loss [dB]')
        plt.legend()
        plt.show()
        
        # CV set learning curve
       # plt.figure()
        plt.subplot(3,2,2)
        plt.plot(self.MSE_cv_dB_epoch, label='validation Loss')
        plt.title('validation Learning Curve')
        plt.xlabel('Step')
        plt.ylabel('Loss [dB]')
        plt.legend()
        plt.show()
        
        # TRain set learning curve - on obs
        #plt.figure()
        plt.subplot(3,2,3)
        plt.plot(self.MSE_train_dB_epoch_obs, label='Training Loss')
        plt.title('Training Learning Curve - on observation')
        plt.xlabel('Step')
        plt.ylabel('Loss [dB]')
        plt.legend()
        plt.show()
        
        # CV set learning curve
        #plt.figure()
        plt.subplot(3,2,4)
        plt.plot(self.MSE_cv_dB_epoch_obs, label='validation Loss')
        plt.title('validation Learning Curve - on observation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss [dB]')
        plt.legend()
        plt.show()
        
        plt.subplot(3,2,5)
        plt.plot(self.MSE_train_dB_epoch_combined_loss, label='Training Loss')
        plt.title('Training Learning Curve of combined loss')
        plt.xlabel('Step')
        plt.ylabel('Loss [dB]')
        plt.legend()
        plt.show()
        
        # CV set learning curve
        plt.subplot(3,2,6)
        plt.plot(self.MSE_cv_dB_epoch_combined_loss, label='validation Loss')
        plt.title('validation Learning Curve of combined loss')
        plt.xlabel('Step')
        plt.ylabel('Loss [dB]')
        plt.legend()
        plt.show()
        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False,\
     randomInit=False,test_init=None,load_model=False,load_model_path=None,\
        test_lengthMask=None ,to_use_noises_for_MSE = False):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device) 
        else:
            self.model = torch.load(path_results+'best-model.pt', map_location=self.device) 

        self.N_T = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        self.MSE_test_linear_arr_obs = torch.zeros([self.N_T])
        self.MSE_test_linear_arr_noise = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, SysModel.m,SysModel.T_test]).to(self.device)
        y_out_test = torch.zeros([self.N_T, SysModel.n,SysModel.T_test]).to(self.device)
        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden_KNet()
        torch.no_grad()

        start = time.time()

        if (randomInit):
            self.model.InitSequence(test_init, SysModel.T_test)               
        else:
            if SysModel.is_m1x_0_test_true:
                self.model.InitSequence(SysModel.m1x_0_test, SysModel.T_test)
            else:    
                self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_T,1,1), SysModel.T_test)         
        
        for t in range(0, SysModel.T_test):
            x_out_test[:,:, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:,:, t],2)))
            y_out_test[:,:, t] = torch.squeeze(self.model.m1y)
        end = time.time()
        t = end - start

        # MSE loss
        for j in range(self.N_T):# cannot use batch due to different length and std computation  
            if(MaskOnState):
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,mask,test_lengthMask[j]], test_target[j,mask,test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,mask,:], test_target[j,mask,:]).item()
            else:
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:2,:], test_target[j,:2,:]).item()
                    self.MSE_test_linear_arr_obs[j] = loss_fn(y_out_test[j,:,:], test_input[j,:,:]).item()
                    if to_use_noises_for_MSE:
                        self.MSE_test_linear_arr_noise[j] = loss_fn(x_out_test[j,2:,:], test_target[j,2:,:]).item() 
        loss_fin = self.MSE_test_linear_arr_obs if self.unsupervised else self.MSE_test_linear_arr
        
        # # Average
        # self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        # self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # # Standard deviation
        # self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # # Confidence interval
        # self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg
        
        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        if to_use_noises_for_MSE:
            self.MSE_test_linear_avg_noise = torch.mean(self.MSE_test_linear_arr_noise)
            self.MSE_test_dB_avg_noise = 10 * torch.log10(self.MSE_test_linear_avg_noise)
            self.MSE_test_linear_std_noise = torch.std(self.MSE_test_linear_arr_noise, unbiased=True)
            self.test_std_dB_noise = 10 * torch.log10(self.MSE_test_linear_std_noise + self.MSE_test_linear_avg_noise) - self.MSE_test_dB_avg_noise
        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg
        
        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)
        if to_use_noises_for_MSE:
            return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t,self.MSE_test_linear_arr_noise, self.MSE_test_linear_avg_noise, self.MSE_test_dB_avg_noise]
        else: 
            return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

    
