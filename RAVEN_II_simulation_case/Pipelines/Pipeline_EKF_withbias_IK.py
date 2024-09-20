

import torch
import torch.nn as nn
import random
import time
from Plot import Plot_extended

class Pipeline_EKF:
    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"
        self.h_nn_optimizer = None

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch  # Number of Samples in Batch
        self.learningRate = args.lr  # Learning Rate
        self.weightDecay = args.wd  # L2 Weight Regularization - Weight Decay
        self.alpha = args.alpha  # Composition loss factor
        self.h_nn_optimizer = torch.optim.Adam(self.model.h_nn.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=False, randomInit=False, cv_init=None, train_init=None, train_lengthMask=None, cv_lengthMask=None):
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        self.MSE_bias_db = torch.zeros([self.N_steps])
        MSE_bias = torch.zeros([self.N_B])

        if MaskOnState:
            mask = torch.tensor([True, False, False])
            if SysModel.m == 2:
                mask = torch.tensor([True, False])

        # Normalize data
        train_input_mean = train_input.mean(dim=(0, 2), keepdim=True)
        train_input_std = train_input.std(dim=(0, 2), keepdim=True)
        train_input = (train_input - train_input_mean) / train_input_std

        cv_input = (cv_input - train_input_mean) / train_input_std

        # Check for NaNs after normalization
        if torch.isnan(train_input).any() or torch.isnan(train_target).any():
            raise ValueError("NaN values found in training input or target data after normalization")
        if torch.isnan(cv_input).any() or torch.isnan(cv_target).any():
            raise ValueError("NaN values found in cv input or target data after normalization")

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_steps):
            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            self.h_nn_optimizer.zero_grad()  # Zero gradients for h_nn
            self.model.train()
            self.model.batch_size = self.N_B
            self.model.init_hidden_KNet()

            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            if self.args.randomLength:
                MSE_train_linear_LOSS = torch.zeros([self.N_B])
                MSE_bias = torch.zeros([self.N_B])
                MSE_cv_linear_LOSS = torch.zeros([self.N_CV])
                MSE_cv_bias = torch.zeros([self.N_CV])

            assert self.N_B <= self.N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:
                if self.args.randomLength:
                    y_training_batch[ii, :, train_lengthMask[index, :]] = train_input[index, :, train_lengthMask[index, :]]
                    train_target_batch[ii, :, train_lengthMask[index, :]] = train_target[index, :, train_lengthMask[index, :]]
                else:
                    y_training_batch[ii, :, :] = train_input[index]
                    train_target_batch[ii, :, :] = train_target[index]
                ii += 1

            if torch.isnan(y_training_batch).any() or torch.isnan(train_target_batch).any():
                raise ValueError("NaN values found in training input or target data")

            if randomInit:
                train_init_batch = torch.empty([self.N_B, SysModel.m, 1]).to(self.device)
                ii = 0
                for index in n_e:
                    train_init_batch[ii, :, 0] = torch.squeeze(train_init[index])
                    ii += 1
                self.model.InitSequence(train_init_batch, SysModel.T)
            else:
                self.model.InitSequence(SysModel.m1x_0[:self.N_B, :].unsqueeze(-1), SysModel.T)

            for t in range(0, SysModel.T):
                x_out_training_batch[:, :, t] = self.model(y_training_batch[:, :, t]).squeeze(-1)

            if torch.isnan(x_out_training_batch).any():
                raise ValueError("NaN values found in model output")

            MSE_trainbatch_linear_LOSS = 0
            if self.args.CompositionLoss:
                y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
                for t in range(SysModel.T):
                    y_hat[:, :, t] = self.model.h_nn(x_out_training_batch[:, :, t])

                if MaskOnState:
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
                                x_out_training_batch[jj, mask, train_lengthMask[index]],
                                train_target_batch[jj, mask, train_lengthMask[index]]) + \
                                (1 - self.alpha) * self.loss_fn(
                                y_hat[jj, mask, train_lengthMask[index]],
                                y_training_batch[jj, mask, train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
                            x_out_training_batch[:, mask, :], train_target_batch[:, mask, :]) + \
                            (1 - self.alpha) * self.loss_fn(y_hat[:, mask, :], y_training_batch[:, mask, :])
                else:
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:
                            MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
                                x_out_training_batch[jj, :3, train_lengthMask[index]],
                                train_target_batch[jj, :3, train_lengthMask[index]]) + \
                                (1 - self.alpha) * self.loss_fn(
                                y_hat[jj, :, train_lengthMask[index]],
                                y_training_batch[jj, :, train_lengthMask[index]])
                            MSE_bias[jj] = self.loss_fn(x_out_training_batch[jj, 4:, train_lengthMask[index]],
                                train_target_batch[jj, 4:, train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
                            x_out_training_batch, train_target_batch) + \
                            (1 - self.alpha) * self.loss_fn(y_hat, y_training_batch)
            else:
                if MaskOnState:
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:
                            MSE_train_linear_LOSS[jj] = self.loss_fn(
                                x_out_training_batch[jj, mask, train_lengthMask[index]],
                                train_target_batch[jj, mask, train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.loss_fn(
                            x_out_training_batch[:, mask, :], train_target_batch[:, mask, :])
                else:
                    if self.args.randomLength:
                        jj = 0
                        for index in n_e:
                            MSE_train_linear_LOSS[jj] = self.loss_fn(
                                x_out_training_batch[jj, :3, train_lengthMask[index]],
                                train_target_batch[jj, :3, train_lengthMask[index]])
                            MSE_bias[jj] = self.loss_fn(x_out_training_batch[jj, 4:, train_lengthMask[index]],
                                train_target_batch[jj, 4:, train_lengthMask[index]])
                            jj += 1
                        MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
                    else:
                        MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

            if torch.isnan(MSE_trainbatch_linear_LOSS).any():
                raise ValueError("NaN values found in training loss")

            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])
            #self.MSE_bias_db[ti] = 10 * torch.log10(torch.mean(MSE_bias))
            self.MSE_bias_db[ti] = (torch.mean(MSE_bias))
            ##################
            ### Optimizing ###
            ##################

            MSE_trainbatch_linear_LOSS.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.h_nn.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.h_nn_optimizer.step()  # Update h_nn

            #################################
            ### Validation Sequence Batch ###
            #################################

            self.model.eval()
            self.model.batch_size = self.N_CV
            self.model.init_hidden_KNet()
            with torch.no_grad():
                SysModel.T_test = cv_input.size()[-1]
                x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)

                if randomInit:
                    if cv_init is None:
                        self.model.InitSequence(SysModel.m1x_0.view(self.N_CV, SysModel.m, 1), SysModel.T_test)
                    else:
                        self.model.InitSequence(cv_init, SysModel.T_test)
                else:
                    self.model.InitSequence(SysModel.m1x_0[:self.N_CV].unsqueeze(-1), SysModel.T_test)

                for t in range(0, SysModel.T_test):
                    x_out_cv_batch[:, :, t] = self.model(cv_input[:, :, t]).squeeze(-1)

                MSE_cvbatch_linear_LOSS = 0
                if MaskOnState:
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(
                                x_out_cv_batch[index, mask, cv_lengthMask[index]],
                                cv_target[index, mask, cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:
                        MSE_cvbatch_linear_LOSS = self.loss_fn(
                            x_out_cv_batch[:, mask, :], cv_target[:, mask, :])
                else:
                    if self.args.randomLength:
                        for index in range(self.N_CV):
                            MSE_cv_linear_LOSS[index] = self.loss_fn(
                                x_out_cv_batch[index, :3, cv_lengthMask[index]],
                                cv_target[index, :3, cv_lengthMask[index]])
                            MSE_cv_bias[index] = self.loss_fn(
                                x_out_cv_batch[index, 4:, cv_lengthMask[index]],
                                cv_target[index, 4:, cv_lengthMask[index]])
                        MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
                    else:
                        MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, path_results + 'best-model.pt')

            print(ti,  "MSE Loss :", self.MSE_cv_dB_epoch[ti], "[dB]")

            if ti > 1:
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False, randomInit=False, test_init=None, load_model=False, load_model_path=None, test_lengthMask=None):
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device)
        else:
            self.model = torch.load(path_results + 'best-model.pt', map_location=self.device)

        self.N_T = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        self.MSE_test_linear_bias = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device)

        if MaskOnState:
            mask = torch.tensor([True, False, False])
            if SysModel.m == 2:
                mask = torch.tensor([True, False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        self.model.init_hidden_KNet()
        torch.no_grad()

        start = time.time()

        if randomInit:
            self.model.InitSequence(test_init, SysModel.T_test)
        else:
            self.model.InitSequence(SysModel.m1x_0[:self.N_T].unsqueeze(-1), SysModel.T_test)
            is_equal = torch.equal(SysModel.m1x_0, test_target[:, :, 0])
            print(f"m1x_0 == test_target[:, :, 0]: {is_equal}")

        for t in range(0, SysModel.T_test):
            x_out_test[:, :, t] = self.model(test_input[:, :, t]).squeeze(-1)

        end = time.time()
        t = end - start

        # MSE loss
        for j in range(self.N_T):
            if MaskOnState:
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(
                        x_out_test[j, mask, test_lengthMask[j]], test_target[j, mask, test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(
                        x_out_test[j, mask, :], test_target[j, mask, :]).item()
            else:
                if self.args.randomLength:
                    self.MSE_test_linear_arr[j] = loss_fn(
                        x_out_test[j, :3, test_lengthMask[j]], test_target[j, :3, test_lengthMask[j]]).item()
                    self.MSE_test_linear_bias[j] = loss_fn(
                        x_out_test[j, 3:, test_lengthMask[j]], test_target[j,3:, test_lengthMask[j]]).item()
                else:
                    self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :3, :], test_target[j, :3, :]).item()
                    self.MSE_test_linear_bias[j] = loss_fn(x_out_test[j, 3:, :], test_target[j, 3:, :]).item()

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        self.MSE_test_bias_db = 10 * torch.log10(torch.mean(self.MSE_test_linear_bias))

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        print(f'bias MSE = {self.MSE_test_bias_db}[db]')
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):
        self.Plot = Plot_extended(self.folderName, self.modelName)
        self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg, self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)
        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)



# import torch
# import torch.nn as nn
# import random
# import time
# from Plot import Plot_extended


# class Pipeline_EKF:
#     def __init__(self, Time, folderName, modelName):
#         super().__init__()
#         self.Time = Time
#         self.folderName = folderName + '/'
#         self.modelName = modelName
#         self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
#         self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"
#         self.h_nn_optimizer = None

#     def save(self):
#         torch.save(self, self.PipelineName)

#     def setssModel(self, ssModel):
#         self.ssModel = ssModel

#     def setModel(self, model):
#         self.model = model

#     def setTrainingParams(self, args):
#         self.args = args
#         if args.use_cuda:
#             self.device = torch.device('cuda')
#         else:
#             self.device = torch.device('cpu')
#         self.N_steps = args.n_steps  # Number of Training Steps
#         self.N_B = args.n_batch # Number of Samples in Batch
#         self.learningRate = args.lr # Learning Rate
#         self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
#         self.alpha = args.alpha # Composition loss factor
#         self.h_nn_optimizer = torch.optim.Adam(self.model.h_nn.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
#         # MSE LOSS Function
#         self.loss_fn = nn.MSELoss(reduction='mean')

#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
    
#     def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=False, randomInit=False, cv_init=None, train_init=None, train_lengthMask=None, cv_lengthMask=None):
#         self.N_E = len(train_input)
#         self.N_CV = len(cv_input)

#         self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
#         self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

#         self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
#         self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        
#         if MaskOnState:
#             mask = torch.tensor([True, False, False])
#             if SysModel.m == 2:
#                 mask = torch.tensor([True, False])

#         # Normalize data
#         train_input_mean = train_input.mean(dim=(0, 2), keepdim=True)
#         train_input_std = train_input.std(dim=(0, 2), keepdim=True)
#         train_input = (train_input - train_input_mean) / train_input_std

#         cv_input = (cv_input - train_input_mean) / train_input_std

#         # Check for NaNs after normalization
#         if torch.isnan(train_input).any() or torch.isnan(train_target).any():
#             raise ValueError("NaN values found in training input or target data after normalization")
#         if torch.isnan(cv_input).any() or torch.isnan(cv_target).any():
#             raise ValueError("NaN values found in cv input or target data after normalization")

#         self.MSE_cv_dB_opt = 1000
#         self.MSE_cv_idx_opt = 0

#         for ti in range(0, self.N_steps):
#             ###############################
#             ### Training Sequence Batch ###
#             ###############################
#             self.optimizer.zero_grad()
#             self.h_nn_optimizer.zero_grad()  # Zero gradients for h_nn
#             self.model.train()
#             self.model.batch_size = self.N_B
#             self.model.init_hidden_KNet()

#             y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
#             train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
#             x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
#             if self.args.randomLength:
#                 MSE_train_linear_LOSS = torch.zeros([self.N_B])
#                 MSE_bias = torch.zeros([self.N_B])
#                 MSE_cv_linear_LOSS = torch.zeros([self.N_CV])
#                 MSE_cv_bias = torch.zeros([self.N_CV])
                

#             assert self.N_B <= self.N_E
#             n_e = random.sample(range(self.N_E), k=self.N_B)
#             ii = 0
#             for index in n_e:
#                 if self.args.randomLength:
#                     y_training_batch[ii, :, train_lengthMask[index, :]] = train_input[index, :, train_lengthMask[index, :]]
#                     train_target_batch[ii, :, train_lengthMask[index, :]] = train_target[index, :, train_lengthMask[index, :]]
#                 else:
#                     y_training_batch[ii, :, :] = train_input[index]
#                     train_target_batch[ii, :, :] = train_target[index]
#                 ii += 1
            
#             if torch.isnan(y_training_batch).any() or torch.isnan(train_target_batch).any():
#                 raise ValueError("NaN values found in training input or target data")

#             if randomInit:
#                 train_init_batch = torch.empty([self.N_B, SysModel.m, 1]).to(self.device)
#                 ii = 0
#                 for index in n_e:
#                     train_init_batch[ii, :, 0] = torch.squeeze(train_init[index])
#                     ii += 1
#                 self.model.InitSequence(train_init_batch, SysModel.T)
#             else:
#                 self.model.InitSequence(SysModel.m1x_0[:self.N_B, :].unsqueeze(-1), SysModel.T)

#             for t in range(0, SysModel.T):
#                 x_out_training_batch[:, :, t] = self.model(y_training_batch[:, :, t]).squeeze(-1)


#             if torch.isnan(x_out_training_batch).any():
#                 raise ValueError("NaN values found in model output")

#             MSE_trainbatch_linear_LOSS = 0
#             if self.args.CompositionLoss:
#                 y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
#                 for t in range(SysModel.T):
#                     y_hat[:, :, t] = self.model.h_nn(x_out_training_batch[:, :, t])

#                 if MaskOnState:
#                     if self.args.randomLength:
#                         jj = 0
#                         for index in n_e:
#                             MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
#                                 x_out_training_batch[jj, mask, train_lengthMask[index]],
#                                 train_target_batch[jj, mask, train_lengthMask[index]]) + \
#                                 (1 - self.alpha) * self.loss_fn(
#                                 y_hat[jj, mask, train_lengthMask[index]],
#                                 y_training_batch[jj, mask, train_lengthMask[index]])
#                             jj += 1
#                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
#                     else:
#                         MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
#                             x_out_training_batch[:, mask, :], train_target_batch[:, mask, :]) + \
#                             (1 - self.alpha) * self.loss_fn(y_hat[:, mask, :], y_training_batch[:, mask, :])
#                 else:
#                     if self.args.randomLength:
#                         jj = 0
#                         for index in n_e:
#                             MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
#                                 x_out_training_batch[jj, :3, train_lengthMask[index]],
#                                 train_target_batch[jj, :3, train_lengthMask[index]]) + \
#                                 (1 - self.alpha) * self.loss_fn(
#                                 y_hat[jj, :, train_lengthMask[index]],
#                                 y_training_batch[jj, :, train_lengthMask[index]])
#                             MSE_bias[jj] = self.loss_fn( x_out_training_batch[jj, 4:, train_lengthMask[index]],
#                               train_target_batch[jj, 4:, train_lengthMask[index]])
#                             jj += 1
#                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
#                     else:
#                         MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
#                             x_out_training_batch, train_target_batch) + \
#                             (1 - self.alpha) * self.loss_fn(y_hat, y_training_batch)
#             else:
#                 if MaskOnState:
#                     if self.args.randomLength:
#                         jj = 0
#                         for index in n_e:
#                             MSE_train_linear_LOSS[jj] = self.loss_fn(
#                                 x_out_training_batch[jj, mask, train_lengthMask[index]],
#                                 train_target_batch[jj, mask, train_lengthMask[index]])
#                             jj += 1
#                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
#                     else:
#                         MSE_trainbatch_linear_LOSS = self.loss_fn(
#                             x_out_training_batch[:, mask, :], train_target_batch[:, mask, :])
#                 else:
#                     if self.args.randomLength:
#                         jj = 0
#                         for index in n_e:
#                             MSE_train_linear_LOSS[jj] = self.loss_fn(
#                                 x_out_training_batch[jj, :3, train_lengthMask[index]],
#                                 train_target_batch[jj, :3, train_lengthMask[index]])
#                             MSE_bias[jj] = self.loss_fn( x_out_training_batch[jj, 4:, train_lengthMask[index]],
#                               train_target_batch[jj, 4:, train_lengthMask[index]])
#                         jj += 1
#                     #MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)

#                     MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

#             if torch.isnan(MSE_trainbatch_linear_LOSS).any():
#                 raise ValueError("NaN values found in training loss")

#             self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
#             self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])
#             self.MSE_bias_db[ti] = 10 * torch.log10(MSE_bias)

#             ##################
#             ### Optimizing ###
#             ##################

#             MSE_trainbatch_linear_LOSS.backward()

#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             torch.nn.utils.clip_grad_norm_(self.model.h_nn.parameters(), max_norm=1.0)

#             self.optimizer.step()
#             self.h_nn_optimizer.step()  # Update h_nn

#             #################################
#             ### Validation Sequence Batch ###
#             #################################

#             self.model.eval()
#             self.model.batch_size = self.N_CV
#             self.model.init_hidden_KNet()
#             with torch.no_grad():
#                 SysModel.T_test = cv_input.size()[-1]
#                 x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)

#                 if randomInit:
#                     if cv_init is None:
#                         self.model.InitSequence(SysModel.m1x_0.view(self.N_CV, SysModel.m, 1), SysModel.T_test)
#                     else:
#                         self.model.InitSequence(cv_init, SysModel.T_test)
#                 else:
#                     self.model.InitSequence(SysModel.m1x_0[:self.N_CV].unsqueeze(-1), SysModel.T_test)

#                 for t in range(0, SysModel.T_test):
#                     x_out_cv_batch[:, :, t] = self.model(cv_input[:, :, t]).squeeze(-1)

#                 MSE_cvbatch_linear_LOSS = 0
#                 if MaskOnState:
#                     if self.args.randomLength:
#                         for index in range(self.N_CV):
#                             MSE_cv_linear_LOSS[index] = self.loss_fn(
#                                 x_out_cv_batch[index, mask, cv_lengthMask[index]],
#                                 cv_target[index, mask, cv_lengthMask[index]])
#                         MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
#                     else:
#                         MSE_cvbatch_linear_LOSS = self.loss_fn(
#                             x_out_cv_batch[:, mask, :], cv_target[:, mask, :])
#                 else:
#                     if self.args.randomLength:
#                         for index in range(self.N_CV):
#                             MSE_cv_linear_LOSS[index] = self.loss_fn(
#                                 x_out_cv_batch[index, :3, cv_lengthMask[index]],
#                                 cv_target[index, :3, cv_lengthMask[index]])
#                             MSE_cv_bias[index] = self.loss_fn(
#                                 x_out_cv_batch[index, 4:, cv_lengthMask[index]],
#                                 cv_target[index, 4:, cv_lengthMask[index]])
#                         MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
#                     else:
#                         MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

#                 self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
#                 self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

#                 if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
#                     self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
#                     self.MSE_cv_idx_opt = ti
#                     torch.save(self.model, path_results + 'best-model.pt')

#             print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti], "[dB]", "MSE LINEAR:", self.MSE_cv_linear_epoch[ti],"MSE bias", MSE_cv_bias[ti])

#             if ti > 1:
#                 d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
#                 d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]

#             print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

#         return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

#     def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False, \
#       randomInit=False, test_init=None, load_model=False, load_model_path=None, \
#         test_lengthMask=None):
#         # Load model
#         if load_model:
#             self.model = torch.load(load_model_path, map_location=self.device)
#         else:
#             self.model = torch.load(path_results + 'best-model.pt', map_location=self.device)

#         self.N_T = test_input.shape[0]
#         SysModel.T_test = test_input.size()[-1]
#         self.MSE_test_linear_arr = torch.zeros([self.N_T])
#         x_out_test = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device)

#         if MaskOnState:
#             mask = torch.tensor([True, False, False])
#             if SysModel.m == 2:
#                 mask = torch.tensor([True, False])

#         # MSE LOSS Function
#         loss_fn = nn.MSELoss(reduction='mean')

#         # Test mode
#         self.model.eval()
#         self.model.batch_size = self.N_T
#         # Init Hidden State
#         self.model.init_hidden_KNet()
#         torch.no_grad()

#         start = time.time()

#         if randomInit:
#             self.model.InitSequence(test_init, SysModel.T_test)
#         else:
#             self.model.InitSequence(SysModel.m1x_0[:self.N_T].unsqueeze(-1), SysModel.T_test)
#             is_equal = torch.equal(SysModel.m1x_0, test_target[:, :, 0])
#             print(f"m1x_0 == test_target[:, :, 0]: {is_equal}")

        
#         for t in range(0, SysModel.T_test):
#             x_out_test[:, :, t] = self.model(test_input[:, :, t]).squeeze(-1)
        
#         end = time.time()
#         t = end - start

#         # MSE loss
#         for j in range(self.N_T):
#             if MaskOnState:
#                 if self.args.randomLength:
#                     self.MSE_test_linear_arr[j] = loss_fn(
#                         x_out_test[j, mask, test_lengthMask[j]], test_target[j, mask, test_lengthMask[j]]).item()
#                 else:
#                     self.MSE_test_linear_arr[j] = loss_fn(
#                         x_out_test[j, mask, :], test_target[j, mask, :]).item()
#             else:
#                 if self.args.randomLength:
#                     self.MSE_test_linear_arr[j] = loss_fn(
#                         x_out_test[j, :3, test_lengthMask[j]], test_target[j, :3, test_lengthMask[j]]).item()
#                     self.MSE_test_linear_bias[j] = loss_fn(
#                         x_out_test[j, 4:, test_lengthMask[j]], test_target[j,4:, test_lengthMask[j]]).item()
#                 else:
#                     self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :3, :], test_target[j, :3, :]).item()
#                     self.MSE_test_linear_bias[j] = loss_fn(x_out_test[j, 4:, :], test_target[j, 4:, :]).item()
        
#         # Average
#         self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
#         self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
#         self.MSE_test_bias_db = 10 * torch.log10(self.MSE_test_linear_bias)

#         # Standard deviation
#         self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

#         # Confidence interval
#         self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

#         # Print MSE and std
#         str = self.modelName + "-" + "MSE Test:"
#         print(str, self.MSE_test_dB_avg, "[dB]")
#         str = self.modelName + "-" + "STD Test:"
#         print(str, self.test_std_dB, "[dB]")
#         print(f'bias MSE = {self.MSE_test_bias_db}')
#         # Print Run Time
#         print("Inference Time:", t)

#         return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

#     def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):
#         self.Plot = Plot_extended(self.folderName, self.modelName)
#         self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg, self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)
#         self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)


# class Pipeline_EKF:
#     def __init__(self, Time, folderName, modelName):
#         super().__init__()
#         self.Time = Time
#         self.folderName = folderName + '/'
#         self.modelName = modelName
#         self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
#         self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"
#         self.h_nn_optimizer = None

#     def save(self):
#         torch.save(self, self.PipelineName)

#     def setssModel(self, ssModel):
#         self.ssModel = ssModel

#     def setModel(self, model):
#         self.model = model

#     def setTrainingParams(self, args):
#         self.args = args
#         if args.use_cuda:
#             self.device = torch.device('cuda')
#         else:
#             self.device = torch.device('cpu')
#         self.N_steps = args.n_steps  # Number of Training Steps
#         self.N_B = args.n_batch # Number of Samples in Batch
#         self.learningRate = args.lr # Learning Rate
#         self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
#         self.alpha = args.alpha # Composition loss factor
#         self.h_nn_optimizer = torch.optim.Adam(self.model.h_nn.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
#         # MSE LOSS Function
#         self.loss_fn = nn.MSELoss(reduction='mean')

#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
    
#     def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=False, randomInit=False, cv_init=None, train_init=None, train_lengthMask=None, cv_lengthMask=None):
#         self.N_E = len(train_input)
#         self.N_CV = len(cv_input)

#         self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
#         self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

#         self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
#         self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        
#         if MaskOnState:
#             mask = torch.tensor([True, False, False])
#             if SysModel.m == 2:
#                 mask = torch.tensor([True, False])

#         # Normalize data
#         train_input_mean = train_input.mean(dim=(0, 2), keepdim=True)
#         train_input_std = train_input.std(dim=(0, 2), keepdim=True)
#         train_input = (train_input - train_input_mean) / train_input_std

#         cv_input = (cv_input - train_input_mean) / train_input_std

#         # Check for NaNs after normalization
#         if torch.isnan(train_input).any() or torch.isnan(train_target).any():
#             raise ValueError("NaN values found in training input or target data after normalization")
#         if torch.isnan(cv_input).any() or torch.isnan(cv_target).any():
#             raise ValueError("NaN values found in cv input or target data after normalization")

#         self.MSE_cv_dB_opt = 1000
#         self.MSE_cv_idx_opt = 0

#         for ti in range(0, self.N_steps):
#             ###############################
#             ### Training Sequence Batch ###
#             ###############################
#             self.optimizer.zero_grad()
#             self.h_nn_optimizer.zero_grad()  # Zero gradients for h_nn
#             self.model.train()
#             self.model.batch_size = self.N_B
#             self.model.init_hidden_KNet()

#             y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
#             train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
#             x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
#             if self.args.randomLength:
#                 MSE_train_linear_LOSS = torch.zeros([self.N_B])
#                 MSE_cv_linear_LOSS = torch.zeros([self.N_CV])

#             assert self.N_B <= self.N_E
#             n_e = random.sample(range(self.N_E), k=self.N_B)
#             ii = 0
#             for index in n_e:
#                 if self.args.randomLength:
#                     y_training_batch[ii, :, train_lengthMask[index, :]] = train_input[index, :, train_lengthMask[index, :]]
#                     train_target_batch[ii, :, train_lengthMask[index, :]] = train_target[index, :, train_lengthMask[index, :]]
#                 else:
#                     y_training_batch[ii, :, :] = train_input[index]
#                     train_target_batch[ii, :, :] = train_target[index]
#                 ii += 1
            
#             if torch.isnan(y_training_batch).any() or torch.isnan(train_target_batch).any():
#                 raise ValueError("NaN values found in training input or target data")

#             if randomInit:
#                 train_init_batch = torch.empty([self.N_B, SysModel.m, 1]).to(self.device)
#                 ii = 0
#                 for index in n_e:
#                     train_init_batch[ii, :, 0] = torch.squeeze(train_init[index])
#                     ii += 1
#                 self.model.InitSequence(train_init_batch, SysModel.T)
#             else:
#                 self.model.InitSequence(SysModel.m1x_0[:self.N_B, :].unsqueeze(-1), SysModel.T)

#             for t in range(0, SysModel.T):
#                 x_out_training_batch[:, :, t] = self.model(y_training_batch[:, :, t])

#             if torch.isnan(x_out_training_batch).any():
#                 raise ValueError("NaN values found in model output")

#             MSE_trainbatch_linear_LOSS = 0
#             if self.args.CompositionLoss:
#                 y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
#                 for t in range(SysModel.T):
#                     y_hat[:, :, t] = self.model.h_nn(x_out_training_batch[:, :, t])

#                 if MaskOnState:
#                     if self.args.randomLength:
#                         jj = 0
#                         for index in n_e:
#                             MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
#                                 x_out_training_batch[jj, mask, train_lengthMask[index]],
#                                 train_target_batch[jj, mask, train_lengthMask[index]]) + \
#                                 (1 - self.alpha) * self.loss_fn(
#                                 y_hat[jj, mask, train_lengthMask[index]],
#                                 y_training_batch[jj, mask, train_lengthMask[index]])
#                             jj += 1
#                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
#                     else:
#                         MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
#                             x_out_training_batch[:, mask, :], train_target_batch[:, mask, :]) + \
#                             (1 - self.alpha) * self.loss_fn(y_hat[:, mask, :], y_training_batch[:, mask, :])
#                 else:
#                     if self.args.randomLength:
#                         jj = 0
#                         for index in n_e:
#                             MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
#                                 x_out_training_batch[jj, :, train_lengthMask[index]],
#                                 train_target_batch[jj, :, train_lengthMask[index]]) + \
#                                 (1 - self.alpha) * self.loss_fn(
#                                 y_hat[jj, :, train_lengthMask[index]],
#                                 y_training_batch[jj, :, train_lengthMask[index]])
#                             jj += 1
#                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
#                     else:
#                         MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
#                             x_out_training_batch, train_target_batch) + \
#                             (1 - self.alpha) * self.loss_fn(y_hat, y_training_batch)
#             else:
#                 if MaskOnState:
#                     if self.args.randomLength:
#                         jj = 0
#                         for index in n_e:
#                             MSE_train_linear_LOSS[jj] = self.loss_fn(
#                                 x_out_training_batch[jj, mask, train_lengthMask[index]],
#                                 train_target_batch[jj, mask, train_lengthMask[index]])
#                             jj += 1
#                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
#                     else:
#                         MSE_trainbatch_linear_LOSS = self.loss_fn(
#                             x_out_training_batch[:, mask, :], train_target_batch[:, mask, :])
#                 else:
#                     if self.args.randomLength:
#                         jj = 0
#                         for index in n_e:
#                             MSE_train_linear_LOSS[jj] = self.loss_fn(
#                                 x_out_training_batch[jj, :, train_lengthMask[index]],
#                                 train_target_batch[jj, :, train_lengthMask[index]])
#                             jj += 1
#                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
#                     else:
#                         MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

#             if torch.isnan(MSE_trainbatch_linear_LOSS).any():
#                 raise ValueError("NaN values found in training loss")

#             self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
#             self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

#             ##################
#             ### Optimizing ###
#             ##################

#             MSE_trainbatch_linear_LOSS.backward()

#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             torch.nn.utils.clip_grad_norm_(self.model.h_nn.parameters(), max_norm=1.0)

#             self.optimizer.step()
#             self.h_nn_optimizer.step()  # Update h_nn

#             #################################
#             ### Validation Sequence Batch ###
#             #################################

#             self.model.eval()
#             self.model.batch_size = self.N_CV
#             self.model.init_hidden_KNet()
#             with torch.no_grad():
#                 SysModel.T_test = cv_input.size()[-1]
#                 x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)

#                 if randomInit:
#                     if cv_init is None:
#                         self.model.InitSequence(SysModel.m1x_0.view(self.N_CV, SysModel.m, 1), SysModel.T_test)
#                     else:
#                         self.model.InitSequence(cv_init, SysModel.T_test)
#                 else:
#                     self.model.InitSequence(SysModel.m1x_0[:self.N_CV].unsqueeze(-1), SysModel.T_test)

#                 for t in range(0, SysModel.T_test):
#                     x_out_cv_batch[:, :, t] = self.model(cv_input[:, :, t])

#                 MSE_cvbatch_linear_LOSS = 0
#                 if MaskOnState:
#                     if self.args.randomLength:
#                         for index in range(self.N_CV):
#                             MSE_cv_linear_LOSS[index] = self.loss_fn(
#                                 x_out_cv_batch[index, mask, cv_lengthMask[index]],
#                                 cv_target[index, mask, cv_lengthMask[index]])
#                         MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
#                     else:
#                         MSE_cvbatch_linear_LOSS = self.loss_fn(
#                             x_out_cv_batch[:, mask, :], cv_target[:, mask, :])
#                 else:
#                     if self.args.randomLength:
#                         for index in range(self.N_CV):
#                             MSE_cv_linear_LOSS[index] = self.loss_fn(
#                                 x_out_cv_batch[index, :, cv_lengthMask[index]],
#                                 cv_target[index, :, cv_lengthMask[index]])
#                         MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
#                     else:
#                         MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

#                 self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
#                 self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

#                 if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
#                     self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
#                     self.MSE_cv_idx_opt = ti
#                     torch.save(self.model, path_results + 'best-model.pt')

#             print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti], "[dB]", "MSE LINEAR:", self.MSE_cv_linear_epoch[ti])

#             if ti > 1:
#                 d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
#                 d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]

#             print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

#         return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

#     def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False, \
#       randomInit=False, test_init=None, load_model=False, load_model_path=None, \
#         test_lengthMask=None):
#         # Load model
#         if load_model:
#             self.model = torch.load(load_model_path, map_location=self.device)
#         else:
#             self.model = torch.load(path_results + 'best-model.pt', map_location=self.device)

#         self.N_T = test_input.shape[0]
#         SysModel.T_test = test_input.size()[-1]
#         self.MSE_test_linear_arr = torch.zeros([self.N_T])
#         x_out_test = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device)

#         if MaskOnState:
#             mask = torch.tensor([True, False, False])
#             if SysModel.m == 2:
#                 mask = torch.tensor([True, False])

#         # MSE LOSS Function
#         loss_fn = nn.MSELoss(reduction='mean')

#         # Test mode
#         self.model.eval()
#         self.model.batch_size = self.N_T
#         # Init Hidden State
#         self.model.init_hidden_KNet()
#         torch.no_grad()

#         start = time.time()

#         if randomInit:
#             self.model.InitSequence(test_init, SysModel.T_test)
#         else:
#             self.model.InitSequence(SysModel.m1x_0[:self.N_T].unsqueeze(-1), SysModel.T_test)

        
#         for t in range(0, SysModel.T_test):
#             x_out_test[:, :, t] = self.model(test_input[:, :, t])
        
#         end = time.time()
#         t = end - start

#         # MSE loss
#         for j in range(self.N_T):
#             if MaskOnState:
#                 if self.args.randomLength:
#                     self.MSE_test_linear_arr[j] = loss_fn(
#                         x_out_test[j, mask, test_lengthMask[j]], test_target[j, mask, test_lengthMask[j]]).item()
#                 else:
#                     self.MSE_test_linear_arr[j] = loss_fn(
#                         x_out_test[j, mask, :], test_target[j, mask, :]).item()
#             else:
#                 if self.args.randomLength:
#                     self.MSE_test_linear_arr[j] = loss_fn(
#                         x_out_test[j, :, test_lengthMask[j]], test_target[j, :, test_lengthMask[j]]).item()
#                 else:
#                     self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :, :], test_target[j, :, :]).item()
        
#         # Average
#         self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
#         self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

#         # Standard deviation
#         self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

#         # Confidence interval
#         self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

#         # Print MSE and std
#         str = self.modelName + "-" + "MSE Test:"
#         print(str, self.MSE_test_dB_avg, "[dB]")
#         str = self.modelName + "-" + "STD Test:"
#         print(str, self.test_std_dB, "[dB]")
#         # Print Run Time
#         print("Inference Time:", t)

#         return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

#     def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

#         self.Plot = Plot_extended(self.folderName, self.modelName)

#         self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
#                                 self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

#         self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

# # import torch
# # import torch.nn as nn
# # import random
# # import time
# # from Plot import Plot_extended

# # class Pipeline_EKF:

# #     def __init__(self, Time, folderName, modelName):
# #         super().__init__()
# #         self.Time = Time
# #         self.folderName = folderName + '/'
# #         self.modelName = modelName
# #         self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
# #         self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

# #     def save(self):
# #         torch.save(self, self.PipelineName)

# #     def setssModel(self, ssModel):
# #         self.ssModel = ssModel

# #     def setModel(self, model):
# #         self.model = model

# #     def setTrainingParams(self, args):
# #         self.args = args
# #         if args.use_cuda:
# #             self.device = torch.device('cuda')
# #         else:
# #             self.device = torch.device('cpu')
# #         self.N_steps = args.n_steps  # Number of Training Steps
# #         self.N_B = args.n_batch # Number of Samples in Batch
# #         self.learningRate = args.lr # Learning Rate
# #         self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
# #         self.alpha = args.alpha # Composition loss factor
# #         # MSE LOSS Function
# #         self.loss_fn = nn.MSELoss(reduction='mean')

# #         # Use the optim package to define an Optimizer that will update the weights of
# #         # the model for us. Here we will use Adam; the optim package contains many other
# #         # optimization algoriths. The first argument to the Adam constructor tells the
# #         # optimizer which Tensors it should update.
# #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
    
# #     def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, \
# #         MaskOnState=False, randomInit=False, cv_init=None, train_init=None, \
# #         train_lengthMask=None, cv_lengthMask=None):

# #         self.N_E = len(train_input)
# #         self.N_CV = len(cv_input)

# #         self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
# #         self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

# #         self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
# #         self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        
# #         if MaskOnState:
# #             mask = torch.tensor([True, False, False])
# #             if SysModel.m == 2:
# #                 mask = torch.tensor([True, False])

# #         # Normalize data
# #         # train_input_mean = train_input.mean(dim=(0, 2), keepdim=True)
# #         # train_input_std = train_input.std(dim=(0, 2), keepdim=True)
# #         # train_input = (train_input - train_input_mean) / train_input_std

# #         train_target_mean = train_target.mean(dim=(0, 2), keepdim=True)
# #         train_target_std = train_target.std(dim=(0, 2), keepdim=True)
# #         train_target = (train_target - train_target_mean) / train_target_std

# #         # cv_input = (cv_input - train_input_mean) / train_input_std
# #         cv_target = (cv_target - train_target_mean) / train_target_std

# #         # Check for NaNs after normalization
# #         if torch.isnan(train_input).any() or torch.isnan(train_target).any():
# #             raise ValueError("NaN values found in training input or target data after normalization")
# #         if torch.isnan(cv_input).any() or torch.isnan(cv_target).any():
# #             raise ValueError("NaN values found in cv input or target data after normalization")

# #         ##############
# #         ### Epochs ###
# #         ##############

# #         self.MSE_cv_dB_opt = 1000
# #         self.MSE_cv_idx_opt = 0

# #         for ti in range(0, self.N_steps):

# #             ###############################
# #             ### Training Sequence Batch ###
# #             ###############################
# #             self.optimizer.zero_grad()
# #             # Training Mode
# #             self.model.train()
# #             self.model.batch_size = self.N_B
# #             # Init Hidden State
# #             self.model.init_hidden_KNet()

# #             # Init Training Batch tensors
# #             y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
# #             train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
# #             x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
# #             if self.args.randomLength:
# #                 MSE_train_linear_LOSS = torch.zeros([self.N_B])
# #                 MSE_cv_linear_LOSS = torch.zeros([self.N_CV])

# #             # Randomly select N_B training sequences
# #             assert self.N_B <= self.N_E # N_B must be smaller than N_E
# #             n_e = random.sample(range(self.N_E), k=self.N_B)
# #             ii = 0
# #             for index in n_e:
# #                 if self.args.randomLength:
# #                     y_training_batch[ii, :, train_lengthMask[index, :]] = train_input[index, :, train_lengthMask[index, :]]
# #                     train_target_batch[ii, :, train_lengthMask[index, :]] = train_target[index, :, train_lengthMask[index, :]]
# #                 else:
# #                     y_training_batch[ii, :, :] = train_input[index]
# #                     train_target_batch[ii, :, :] = train_target[index]
# #                 ii += 1
            
# #             # Check for NaNs in input data
# #             if torch.isnan(y_training_batch).any() or torch.isnan(train_target_batch).any():
# #                 raise ValueError("NaN values found in training input or target data")

# #             # Print intermediate values
# #             print("Training Input Batch:", y_training_batch)
# #             print("Training Target Batch:", train_target_batch)

# #             # Init Sequence
# #             if randomInit:
# #                 train_init_batch = torch.empty([self.N_B, SysModel.m, 1]).to(self.device)
# #                 ii = 0
# #                 for index in n_e:
# #                     train_init_batch[ii, :, 0] = torch.squeeze(train_init[index])
# #                     ii += 1
# #                 self.model.InitSequence(train_init_batch, SysModel.T)
# #             else:
# #                 self.model.InitSequence(SysModel.m1x_0[:self.N_B, :].unsqueeze(-1), SysModel.T)

# #             # Forward Computation
# #             for t in range(0, SysModel.T):
# #                 x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t], 2)))

# #             # Check for NaNs in output data
# #             if torch.isnan(x_out_training_batch).any():
# #                 raise ValueError("NaN values found in model output")

# #             # Print intermediate values
# #             print("Model Output Batch:", x_out_training_batch)

# #             # Compute Training Loss
# #             MSE_trainbatch_linear_LOSS = 0
# #             if self.args.CompositionLoss:
# #                 y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
# #                 for t in range(SysModel.T):
# #                     y_hat[:, :, t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_training_batch[:, :, t])))

# #                 if MaskOnState:
# #                     if self.args.randomLength:
# #                         jj = 0
# #                         for index in n_e:
# #                             MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
# #                                 x_out_training_batch[jj, mask, train_lengthMask[index]],
# #                                 train_target_batch[jj, mask, train_lengthMask[index]]) + \
# #                                 (1 - self.alpha) * self.loss_fn(
# #                                 y_hat[jj, mask, train_lengthMask[index]],
# #                                 y_training_batch[jj, mask, train_lengthMask[index]])
# #                             jj += 1
# #                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
# #                     else:
# #                         MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
# #                             x_out_training_batch[:, mask, :], train_target_batch[:, mask, :]) + \
# #                             (1 - self.alpha) * self.loss_fn(y_hat[:, mask, :], y_training_batch[:, mask, :])
# #                 else:
# #                     if self.args.randomLength:
# #                         jj = 0
# #                         for index in n_e:
# #                             MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
# #                                 x_out_training_batch[jj, :, train_lengthMask[index]],
# #                                 train_target_batch[jj, :, train_lengthMask[index]]) + \
# #                                 (1 - self.alpha) * self.loss_fn(
# #                                 y_hat[jj, :, train_lengthMask[index]],
# #                                 y_training_batch[jj, :, train_lengthMask[index]])
# #                             jj += 1
# #                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
# #                     else:
# #                         MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
# #                             x_out_training_batch, train_target_batch) + \
# #                             (1 - self.alpha) * self.loss_fn(y_hat, y_training_batch)
# #             else:
# #                 if MaskOnState:
# #                     if self.args.randomLength:
# #                         jj = 0
# #                         for index in n_e:
# #                             MSE_train_linear_LOSS[jj] = self.loss_fn(
# #                                 x_out_training_batch[jj, mask, train_lengthMask[index]],
# #                                 train_target_batch[jj, mask, train_lengthMask[index]])
# #                             jj += 1
# #                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
# #                     else:
# #                         MSE_trainbatch_linear_LOSS = self.loss_fn(
# #                             x_out_training_batch[:, mask, :], train_target_batch[:, mask, :])
# #                 else:
# #                     if self.args.randomLength:
# #                         jj = 0
# #                         for index in n_e:
# #                             MSE_train_linear_LOSS[jj] = self.loss_fn(
# #                                 x_out_training_batch[jj, :, train_lengthMask[index]],
# #                                 train_target_batch[jj, :, train_lengthMask[index]])
# #                             jj += 1
# #                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
# #                     else:
# #                         MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

# #             # Check for NaNs in loss
# #             if torch.isnan(MSE_trainbatch_linear_LOSS).any():
# #                 raise ValueError("NaN values found in training loss")

# #             # Print intermediate values
# #             print("Training Loss:", MSE_trainbatch_linear_LOSS)

# #             # dB Loss
# #             self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
# #             self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

# #             ##################
# #             ### Optimizing ###
# #             ##################

# #             MSE_trainbatch_linear_LOSS.backward()

# #             # Clip gradients to prevent NaNs
# #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

# #             self.optimizer.step()

# #             #################################
# #             ### Validation Sequence Batch ###
# #             #################################

# #             # Cross Validation Mode
# #             self.model.eval()
# #             self.model.batch_size = self.N_CV
# #             # Init Hidden State
# #             self.model.init_hidden_KNet()
# #             with torch.no_grad():

# #                 SysModel.T_test = cv_input.size()[-1] # T_test is the maximum length of the CV sequences

# #                 x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)
                
# #                 # Init Sequence
# #                 if randomInit:
# #                     if cv_init is None:
# #                         print(f'M1x size={SysModel.m1x_0.shape}')
# #                         self.model.InitSequence(SysModel.m1x_0.view(self.N_CV, SysModel.m, 1), SysModel.T_test)
# #                     else:
# #                         self.model.InitSequence(cv_init, SysModel.T_test)
# #                 else:
# #                     self.model.InitSequence(SysModel.m1x_0[:self.N_CV].unsqueeze(-1), SysModel.T_test)

# #                 for t in range(0, SysModel.T_test):
# #                     x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input[:, :, t], 2)))

# #                 # Compute CV Loss
# #                 MSE_cvbatch_linear_LOSS = 0
# #                 if MaskOnState:
# #                     if self.args.randomLength:
# #                         for index in range(self.N_CV):
# #                             MSE_cv_linear_LOSS[index] = self.loss_fn(
# #                                 x_out_cv_batch[index, mask, cv_lengthMask[index]],
# #                                 cv_target[index, mask, cv_lengthMask[index]])
# #                         MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
# #                     else:
# #                         MSE_cvbatch_linear_LOSS = self.loss_fn(
# #                             x_out_cv_batch[:, mask, :], cv_target[:, mask, :])
# #                 else:
# #                     if self.args.randomLength:
# #                         for index in range(self.N_CV):
# #                             MSE_cv_linear_LOSS[index] = self.loss_fn(
# #                                 x_out_cv_batch[index, :, cv_lengthMask[index]],
# #                                 cv_target[index, :, cv_lengthMask[index]])
# #                         MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
# #                     else:
# #                         MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

# #                 # dB Loss
# #                 self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
# #                 self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

# #                 if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
# #                     self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
# #                     self.MSE_cv_idx_opt = ti
# #                     torch.save(self.model, path_results + 'best-model.pt')

# #             ########################
# #             ### Training Summary ###
# #             ########################
# #             print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
# #                   "[dB]", "MSE LINEAR:", self.MSE_cv_linear_epoch[ti])

# #             if ti > 1:
# #                 d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
# #                 d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
# #                 print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

# #             print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

# #         return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

# #     def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False, \
# #       randomInit=False, test_init=None, load_model=False, load_model_path=None, \
# #         test_lengthMask=None):
# #         # Load model
# #         if load_model:
# #             self.model = torch.load(load_model_path, map_location=self.device)
# #         else:
# #             self.model = torch.load(path_results + 'best-model.pt', map_location=self.device)

# #         self.N_T = test_input.shape[0]
# #         SysModel.T_test = test_input.size()[-1]
# #         self.MSE_test_linear_arr = torch.zeros([self.N_T])
# #         x_out_test = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device)

# #         if MaskOnState:
# #             mask = torch.tensor([True, False, False])
# #             if SysModel.m == 2:
# #                 mask = torch.tensor([True, False])

# #         # MSE LOSS Function
# #         loss_fn = nn.MSELoss(reduction='mean')

# #         # Test mode
# #         self.model.eval()
# #         self.model.batch_size = self.N_T
# #         # Init Hidden State
# #         self.model.init_hidden_KNet()
# #         torch.no_grad()

# #         start = time.time()

# #         if randomInit:
# #             self.model.InitSequence(test_init, SysModel.T_test)
# #         else:
# #             self.model.InitSequence(SysModel.m1x_0[:self.N_T].unsqueeze(-1), SysModel.T_test)

        
# #         for t in range(0, SysModel.T_test):
# #             x_out_test[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:, :, t], 2)))
        
# #         end = time.time()
# #         t = end - start

# #         # MSE loss
# #         for j in range(self.N_T):
# #             if MaskOnState:
# #                 if self.args.randomLength:
# #                     self.MSE_test_linear_arr[j] = loss_fn(
# #                         x_out_test[j, mask, test_lengthMask[j]], test_target[j, mask, test_lengthMask[j]]).item()
# #                 else:
# #                     self.MSE_test_linear_arr[j] = loss_fn(
# #                         x_out_test[j, mask, :], test_target[j, mask, :]).item()
# #             else:
# #                 if self.args.randomLength:
# #                     self.MSE_test_linear_arr[j] = loss_fn(
# #                         x_out_test[j, :, test_lengthMask[j]], test_target[j, :, test_lengthMask[j]]).item()
# #                 else:
# #                     self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :, :], test_target[j, :, :]).item()
        
# #         # Average
# #         self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
# #         self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

# #         # Standard deviation
# #         self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

# #         # Confidence interval
# #         self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

# #         # Print MSE and std
# #         str = self.modelName + "-" + "MSE Test:"
# #         print(str, self.MSE_test_dB_avg, "[dB]")
# #         str = self.modelName + "-" + "STD Test:"
# #         print(str, self.test_std_dB, "[dB]")
# #         # Print Run Time
# #         print("Inference Time:", t)

# #         return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

# #     def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

# #         self.Plot = Plot_extended(self.folderName, self.modelName)

# #         self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
# #                                 self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

# #         self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)


# # import torch
# # import torch.nn as nn
# # import random
# # import time
# # from Plot import Plot_extended


# # class Pipeline_EKF:

# #     def __init__(self, Time, folderName, modelName):
# #         super().__init__()
# #         self.Time = Time
# #         self.folderName = folderName + '/'
# #         self.modelName = modelName
# #         self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
# #         self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

# #     def save(self):
# #         torch.save(self, self.PipelineName)

# #     def setssModel(self, ssModel):
# #         self.ssModel = ssModel

# #     def setModel(self, model):
# #         self.model = model

# #     def setTrainingParams(self, args):
# #         self.args = args
# #         if args.use_cuda:
# #             self.device = torch.device('cuda')
# #         else:
# #             self.device = torch.device('cpu')
# #         self.N_steps = args.n_steps  # Number of Training Steps
# #         self.N_B = args.n_batch # Number of Samples in Batch
# #         self.learningRate = args.lr # Learning Rate
# #         self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
# #         self.alpha = args.alpha # Composition loss factor
# #         # MSE LOSS Function
# #         self.loss_fn = nn.MSELoss(reduction='mean')

# #         # Use the optim package to define an Optimizer that will update the weights of
# #         # the model for us. Here we will use Adam; the optim package contains many other
# #         # optimization algoriths. The first argument to the Adam constructor tells the
# #         # optimizer which Tensors it should update.
# #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

# #     def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, \
                
                        
# #         MaskOnState=False, randomInit=False, cv_init=None, train_init=None, \
# #         train_lengthMask=None, cv_lengthMask=None):

# #         self.N_E = len(train_input)
# #         self.N_CV = len(cv_input)

# #         self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
# #         self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

# #         self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
# #         self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        
# #         if MaskOnState:
# #             mask = torch.tensor([True, False, False])
# #             if SysModel.m == 2:
# #                 mask = torch.tensor([True, False])

# #         # Normalize data
# #         train_input_mean = train_input.mean(dim=(0, 2), keepdim=True)
# #         train_input_std = train_input.std(dim=(0, 2), keepdim=True)
# #         train_input = (train_input - train_input_mean) / train_input_std

# #         train_target_mean = train_target.mean(dim=(0, 2), keepdim=True)
# #         train_target_std = train_target.std(dim=(0, 2), keepdim=True)
# #         train_target = (train_target - train_target_mean) / train_target_std

# #         cv_input = (cv_input - train_input_mean) / train_input_std
# #         cv_target = (cv_target - train_target_mean) / train_target_std
        
# #         if torch.isnan(train_target).any() or torch.isnan(train_input).any():
# #             raise ValueError("NaN values found in training input or target data")
# #         if torch.isnan(cv_target).any() or torch.isnan(cv_input).any():
# #             raise ValueError("NaN values found in validation input or target data")

# #         ##############
# #         ### Epochs ###
# #         ##############

# #         self.MSE_cv_dB_opt = 1000
# #         self.MSE_cv_idx_opt = 0

# #         for ti in range(0, self.N_steps):

# #             ###############################
# #             ### Training Sequence Batch ###
# #             ###############################
# #             self.optimizer.zero_grad()
# #             # Training Mode
# #             self.model.train()
# #             self.model.batch_size = self.N_B
# #             # Init Hidden State
# #             self.model.init_hidden_KNet()

# #             # Init Training Batch tensors
# #             y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
# #             train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
# #             x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
# #             if self.args.randomLength:
# #                 MSE_train_linear_LOSS = torch.zeros([self.N_B])
# #                 MSE_cv_linear_LOSS = torch.zeros([self.N_CV])

# #             # Randomly select N_B training sequences
# #             assert self.N_B <= self.N_E # N_B must be smaller than N_E
# #             n_e = random.sample(range(self.N_E), k=self.N_B)
# #             ii = 0
# #             for index in n_e:
# #                 if self.args.randomLength:
# #                     y_training_batch[ii, :, train_lengthMask[index, :]] = train_input[index, :, train_lengthMask[index, :]]
# #                     train_target_batch[ii, :, train_lengthMask[index, :]] = train_target[index, :, train_lengthMask[index, :]]
# #                 else:
# #                     y_training_batch[ii, :, :] = train_input[index]
# #                     train_target_batch[ii, :, :] = train_target[index]
# #                 ii += 1
            
# #             # Check for NaNs in input data
# #             if torch.isnan(y_training_batch).any() or torch.isnan(train_target_batch).any():
# #                 raise ValueError("NaN values found in training input or target data")

# #             # Init Sequence
# #             if randomInit:
# #                 train_init_batch = torch.empty([self.N_B, SysModel.m, 1]).to(self.device)
# #                 ii = 0
# #                 for index in n_e:
# #                     train_init_batch[ii, :, 0] = torch.squeeze(train_init[index])
# #                     ii += 1
# #                 self.model.InitSequence(train_init_batch, SysModel.T)
# #                 ######M1x0######
# #             else:
# #                 print(f'm1x_0 shape is: {SysModel.m1x_0.shape}')
# #                 # if ti==0:
# #                 #SysModel.m1x_0=SysModel.m1x_0[ti,:]
# #                 # else:
# #                 #     SysModel.m1x_0=SysModel.m1x_0[:]
# #                 self.model.InitSequence(SysModel.m1x_0[:self.N_B, :].unsqueeze(-1), SysModel.T)

            
# #             # Forward Computation
# #             for t in range(0, SysModel.T):
# #                 x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t], 2)))
            
# #             # Compute Training Loss
# #             MSE_trainbatch_linear_LOSS = 0
# #             if self.args.CompositionLoss:
# #                 y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
# #                 for t in range(SysModel.T):
# #                     y_hat[:, :, t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_training_batch[:, :, t])))

# #                 if MaskOnState:
# #                     if self.args.randomLength:
# #                         jj = 0
# #                         for index in n_e:
# #                             MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
# #                                 x_out_training_batch[jj, mask, train_lengthMask[index]],
# #                                 train_target_batch[jj, mask, train_lengthMask[index]]) + \
# #                                 (1 - self.alpha) * self.loss_fn(
# #                                 y_hat[jj, mask, train_lengthMask[index]],
# #                                 y_training_batch[jj, mask, train_lengthMask[index]])
# #                             jj += 1
# #                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
# #                     else:
# #                         MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
# #                             x_out_training_batch[:, mask, :], train_target_batch[:, mask, :]) + \
# #                             (1 - self.alpha) * self.loss_fn(y_hat[:, mask, :], y_training_batch[:, mask, :])
# #                 else:
# #                     if self.args.randomLength:
# #                         jj = 0
# #                         for index in n_e:
# #                             MSE_train_linear_LOSS[jj] = self.alpha * self.loss_fn(
# #                                 x_out_training_batch[jj, :, train_lengthMask[index]],
# #                                 train_target_batch[jj, :, train_lengthMask[index]]) + \
# #                                 (1 - self.alpha) * self.loss_fn(
# #                                 y_hat[jj, :, train_lengthMask[index]],
# #                                 y_training_batch[jj, :, train_lengthMask[index]])
# #                             jj += 1
# #                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
# #                     else:
# #                         MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(
# #                             x_out_training_batch, train_target_batch) + \
# #                             (1 - self.alpha) * self.loss_fn(y_hat, y_training_batch)
# #             else:
# #                 if MaskOnState:
# #                     if self.args.randomLength:
# #                         jj = 0
# #                         for index in n_e:
# #                             MSE_train_linear_LOSS[jj] = self.loss_fn(
# #                                 x_out_training_batch[jj, mask, train_lengthMask[index]],
# #                                 train_target_batch[jj, mask, train_lengthMask[index]])
# #                             jj += 1
# #                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
# #                     else:
# #                         MSE_trainbatch_linear_LOSS = self.loss_fn(
# #                             x_out_training_batch[:, mask, :], train_target_batch[:, mask, :])
# #                 else:
# #                     if self.args.randomLength:
# #                         jj = 0
# #                         for index in n_e:
# #                             MSE_train_linear_LOSS[jj] = self.loss_fn(
# #                                 x_out_training_batch[jj, :, train_lengthMask[index]],
# #                                 train_target_batch[jj, :, train_lengthMask[index]])
# #                             jj += 1
# #                         MSE_trainbatch_linear_LOSS = torch.mean(MSE_train_linear_LOSS)
# #                     else:
# #                         MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

# #             # dB Loss
# #             self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
# #             self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

# #             ##################
# #             ### Optimizing ###
# #             ##################

# #             MSE_trainbatch_linear_LOSS.backward()

# #             # Clip gradients to prevent NaNs
# #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

# #             self.optimizer.step()

# #             #################################
# #             ### Validation Sequence Batch ###
# #             #################################

# #             # Cross Validation Mode
# #             self.model.eval()
# #             self.model.batch_size = self.N_CV
# #             # Init Hidden State
# #             self.model.init_hidden_KNet()
# #             with torch.no_grad():

# #                 SysModel.T_test = cv_input.size()[-1] # T_test is the maximum length of the CV sequences

# #                 x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)
                
# #                 # Init Sequence
# #                 if randomInit:
# #                     if cv_init is None:
# #                         print(f'M1x size={SysModel.m1x_0.shape}')
# #                         self.model.InitSequence(SysModel.m1x_0.view(self.N_CV, SysModel.m, 1), SysModel.T_test)

# #                     else:
# #                         self.model.InitSequence(cv_init, SysModel.T_test)
# #                 else:
                    
# #                     self.model.InitSequence(SysModel.m1x_0[:self.N_CV].unsqueeze(-1), SysModel.T_test)


# #                 for t in range(0, SysModel.T_test):
# #                     x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input[:, :, t], 2)))
                
# #                 # Compute CV Loss
# #                 MSE_cvbatch_linear_LOSS = 0
# #                 if MaskOnState:
# #                     if self.args.randomLength:
# #                         for index in range(self.N_CV):
# #                             MSE_cv_linear_LOSS[index] = self.loss_fn(
# #                                 x_out_cv_batch[index, mask, cv_lengthMask[index]],
# #                                 cv_target[index, mask, cv_lengthMask[index]])
# #                         MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
# #                     else:
# #                         MSE_cvbatch_linear_LOSS = self.loss_fn(
# #                             x_out_cv_batch[:, mask, :], cv_target[:, mask, :])
# #                 else:
# #                     if self.args.randomLength:
# #                         for index in range(self.N_CV):
# #                             MSE_cv_linear_LOSS[index] = self.loss_fn(
# #                                 x_out_cv_batch[index, :, cv_lengthMask[index]],
# #                                 cv_target[index, :, cv_lengthMask[index]])
# #                         MSE_cvbatch_linear_LOSS = torch.mean(MSE_cv_linear_LOSS)
# #                     else:
# #                         MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

# #                 # dB Loss
# #                 self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
# #                 self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                
# #                 if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
# #                     self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
# #                     self.MSE_cv_idx_opt = ti
                    
# #                     torch.save(self.model, path_results + 'best-model.pt')

# #             ########################
# #             ### Training Summary ###
# #             ########################
# #             print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
# #                   "[dB]", "MSE LINEAR:",self.MSE_cv_linear_epoch[ti] )
                      
# #             if ti > 1:
# #                 d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
# #                 d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
# #                 print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

# #             print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

# #         return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

# #     def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False, \
# #       randomInit=False, test_init=None, load_model=False, load_model_path=None, \
# #         test_lengthMask=None):
# #         # Load model
# #         if load_model:
# #             self.model = torch.load(load_model_path, map_location=self.device)
# #         else:
# #             self.model = torch.load(path_results + 'best-model.pt', map_location=self.device)

# #         self.N_T = test_input.shape[0]
# #         SysModel.T_test = test_input.size()[-1]
# #         self.MSE_test_linear_arr = torch.zeros([self.N_T])
# #         x_out_test = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device)

# #         if MaskOnState:
# #             mask = torch.tensor([True, False, False])
# #             if SysModel.m == 2:
# #                 mask = torch.tensor([True, False])

# #         # MSE LOSS Function
# #         loss_fn = nn.MSELoss(reduction='mean')

# #         # Test mode
# #         self.model.eval()
# #         self.model.batch_size = self.N_T
# #         # Init Hidden State
# #         self.model.init_hidden_KNet()
# #         torch.no_grad()

# #         start = time.time()

# #         if randomInit:
# #             self.model.InitSequence(test_init, SysModel.T_test)
# #         else:
# #             self.model.InitSequence(SysModel.m1x_0[:self.N_T].unsqueeze(-1), SysModel.T_test)

        
# #         for t in range(0, SysModel.T_test):
# #             x_out_test[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:, :, t], 2)))
        
# #         end = time.time()
# #         t = end - start

# #         # MSE loss
# #         for j in range(self.N_T):
# #             if MaskOnState:
# #                 if self.args.randomLength:
# #                     self.MSE_test_linear_arr[j] = loss_fn(
# #                         x_out_test[j, mask, test_lengthMask[j]], test_target[j, mask, test_lengthMask[j]]).item()
# #                 else:
# #                     self.MSE_test_linear_arr[j] = loss_fn(
# #                         x_out_test[j, mask, :], test_target[j, mask, :]).item()
# #             else:
# #                 if self.args.randomLength:
# #                     self.MSE_test_linear_arr[j] = loss_fn(
# #                         x_out_test[j, :, test_lengthMask[j]], test_target[j, :, test_lengthMask[j]]).item()
# #                 else:
# #                     self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :, :], test_target[j, :, :]).item()
        
# #         # Average
# #         self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
# #         self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

# #         # Standard deviation
# #         self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

# #         # Confidence interval
# #         self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

# #         # Print MSE and std
# #         str = self.modelName + "-" + "MSE Test:"
# #         print(str, self.MSE_test_dB_avg, "[dB]")
# #         str = self.modelName + "-" + "STD Test:"
# #         print(str, self.test_std_dB, "[dB]")
# #         # Print Run Time
# #         print("Inference Time:", t)

# #         return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]

# #     def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

# #         self.Plot = Plot_extended(self.folderName, self.modelName)

# #         self.Plot.NNPlot_epochs(self.N_steps, MSE_KF_dB_avg,
# #                                 self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

# #         self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)
