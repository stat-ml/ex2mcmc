import random

import numpy as np
import torch
import torch.nn as nn


class PMF:
    def __init__(
        self,
        num_feat=10,
        epsilon=1,
        _lambda=0.1,
        momentum=0.8,
        maxepoch=20,
        num_batches=10,
        batch_size=1000,
    ):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        # Number of batches in each epoch (for SGD optimization),
        self.num_batches = num_batches
        # Number of training samples used in each batches (for SGD optimization)
        self.batch_size = batch_size

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec, test_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值

        pairs_train = train_vec.shape[0]  # traindata 中条目数
        pairs_test = test_vec.shape[0]  # testdata中条目数

        # 1-p-i, 2-m-c
        num_user = (
            int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1
        )  # 第0列，user总数
        num_item = (
            int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1
        )  # 第1列，movie总数

        incremental = False  # 增量
        if (not incremental) or (self.w_Item is None):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(
                num_item,
                self.num_feat,
            )  # numpy.random.randn 电影 M x D 正态分布矩阵
            self.w_User = 0.1 * np.random.randn(
                num_user,
                self.num_feat,
            )  # numpy.random.randn 用户 N x D 正态分布矩阵

            self.w_Item_inc = np.zeros(
                (num_item, self.num_feat),
            )  # 创建电影 M x D 0矩阵
            self.w_User_inc = np.zeros(
                (num_user, self.num_feat),
            )  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  # 检查迭代次数
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  # 根据记录数创建等差array
            np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱

            # Batch update
            for batch in range(self.num_batches):  # 每次迭代要使用的数据量
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                test = np.arange(
                    self.batch_size * batch,
                    self.batch_size * (batch + 1),
                )
                batch_idx = np.mod(
                    test,
                    shuffled_order.shape[0],
                )  # 本次迭代要使用的索引下标

                batch_UserID = np.array(
                    train_vec[shuffled_order[batch_idx], 0],
                    dtype="int32",
                )
                batch_ItemID = np.array(
                    train_vec[shuffled_order[batch_idx], 1],
                    dtype="int32",
                )

                # Compute Objective Function
                pred_out = np.sum(
                    np.multiply(
                        self.w_User[batch_UserID, :],
                        self.w_Item[batch_ItemID, :],
                    ),
                    axis=1,
                )  # mean_inv subtracted # np.multiply对应位置元素相乘

                rawErr = (
                    pred_out
                    - train_vec[shuffled_order[batch_idx], 2]
                    + self.mean_inv
                )

                # Compute gradients
                Ix_User = (
                    2
                    * np.multiply(
                        rawErr[:, np.newaxis],
                        self.w_Item[batch_ItemID, :],
                    )
                    + self._lambda * self.w_User[batch_UserID, :]
                )
                Ix_Item = 2 * np.multiply(
                    rawErr[:, np.newaxis],
                    self.w_User[batch_UserID, :],
                ) + self._lambda * (
                    self.w_Item[batch_ItemID, :]
                )  # np.newaxis :increase the dimension

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = (
                    self.momentum * self.w_Item_inc
                    + self.epsilon * dw_Item / self.batch_size
                )
                self.w_User_inc = (
                    self.momentum * self.w_User_inc
                    + self.epsilon * dw_User / self.batch_size
                )

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(
                        np.multiply(
                            self.w_User[
                                np.array(train_vec[:, 0], dtype="int32"), :
                            ],
                            self.w_Item[
                                np.array(train_vec[:, 1], dtype="int32"), :
                            ],
                        ),
                        axis=1,
                    )  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = np.linalg.norm(rawErr) ** 2 + 0.5 * self._lambda * (
                        np.linalg.norm(self.w_User) ** 2
                        + np.linalg.norm(self.w_Item) ** 2
                    )

                    self.rmse_train.append(np.sqrt(obj / pairs_train))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(
                        np.multiply(
                            self.w_User[
                                np.array(test_vec[:, 0], dtype="int32"), :
                            ],
                            self.w_Item[
                                np.array(test_vec[:, 1], dtype="int32"), :
                            ],
                        ),
                        axis=1,
                    )  # mean_inv subtracted
                    rawErr = pred_out - test_vec[:, 2] + self.mean_inv
                    self.rmse_test.append(
                        np.linalg.norm(rawErr) / np.sqrt(pairs_test),
                    )

                    # Print info
                    if batch == self.num_batches - 1:
                        print(
                            "Epoch: %d, Training RMSE: %f, Test RMSE %f"
                            % (
                                self.epoch,
                                self.rmse_train[-1],
                                self.rmse_test[-1],
                            ),
                        )

    def predict(self, invID):
        return (
            np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv
        )  # numpy.dot 点乘

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv))[
                    -k:
                ]  # numpy.argsort索引排序

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = (
                    intersection_cnt.get(test_vec[i, 0], 0) + 1
                )
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype="int32"))

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(
                invPairs_cnt[int(inv)],
            )

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)


class PMF_torch:
    def __init__(
        self,
        num_feat=10,
        epsilon=1,
        _lambda=0.1,
        momentum=0.8,
        maxepoch=20,
        num_batches=10,
        batch_size=1000,
    ):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        # Number of batches in each epoch (for SGD optimization),
        self.num_batches = num_batches
        # Number of training samples used in each batches (for SGD optimization)
        self.batch_size = batch_size

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []

    def loss(self, params, ratings, partition, use_regularize=True):
        pred_out = torch.sum(
            torch.multiply(params[:partition], params[partition:]),
            axis=1,
        )  # mean_inv subtracted # np.multiply对应位置元素相乘

        rawErr = pred_out - ratings + self.mean_inv

        loss = torch.norm(rawErr) ** 2
        if use_regularize:
            loss += 0.5 * self._lambda * (torch.norm(params) ** 2)
        return loss

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec, test_vec):
        # mean subtraction
        self.mean_inv = torch.mean(train_vec[:, 2])  # 评分平均值

        pairs_train = train_vec.shape[0]  # traindata 中条目数
        pairs_test = test_vec.shape[0]  # testdata中条目数

        # 1-p-i, 2-m-c
        num_user = (
            int(max(torch.amax(train_vec[:, 0]), torch.amax(test_vec[:, 0])))
            + 1
        )  # 第0列，user总数
        num_item = (
            int(max(torch.amax(train_vec[:, 1]), torch.amax(test_vec[:, 1])))
            + 1
        )  # 第1列，movie总数

        incremental = False  # 增量
        if (not incremental) or (self.w_Item is None):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * torch.randn(
                num_item,
                self.num_feat,
            )  # numpy.random.randn 电影 M x D 正态分布矩阵
            self.w_User = 0.1 * torch.randn(
                num_user,
                self.num_feat,
            )  # numpy.random.randn 用户 N x D 正态分布矩阵

            self.w_Item_inc = torch.zeros(
                (num_item, self.num_feat),
            )  # 创建电影 M x D 0矩阵
            self.w_User_inc = torch.zeros(
                (num_user, self.num_feat),
            )  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  # 检查迭代次数
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  # 根据记录数创建等差array
            np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱

            # Batch update
            for batch in range(self.num_batches):  # 每次迭代要使用的数据量
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                test = np.arange(
                    self.batch_size * batch,
                    self.batch_size * (batch + 1),
                )
                batch_idx = np.mod(
                    test,
                    shuffled_order.shape[0],
                )  # 本次迭代要使用的索引下标

                batch_UserID = np.array(
                    train_vec[shuffled_order[batch_idx], 0],
                    dtype="int32",
                )
                batch_ItemID = np.array(
                    train_vec[shuffled_order[batch_idx], 1],
                    dtype="int32",
                )

                w_User_batch = self.w_User[batch_UserID, :]
                w_Item_batch = self.w_Item[batch_ItemID, :]
                concat_shape = w_User_batch.shape[0]

                w_big = torch.cat(
                    (w_User_batch.detach(), w_Item_batch.detach()),
                ).requires_grad_()

                # Compute gradients
                ratings = train_vec[shuffled_order[batch_idx], 2]
                loss = self.loss(w_big, ratings, concat_shape)

                Ix_w_big = torch.autograd.grad(loss, w_big)[0]
                Ix_User = Ix_w_big[:concat_shape]
                Ix_Item = Ix_w_big[concat_shape:]

                # Ix_User = 2 * torch.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                #       + self._lambda * self.w_User[batch_UserID, :]
                # Ix_Item = 2 * torch.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                #       + self._lambda * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = torch.zeros((num_item, self.num_feat))
                dw_User = torch.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = (
                    self.momentum * self.w_Item_inc
                    + self.epsilon * dw_Item / self.batch_size
                )
                self.w_User_inc = (
                    self.momentum * self.w_User_inc
                    + self.epsilon * dw_User / self.batch_size
                )

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    ratings = train_vec[:, 2]
                    concat_shape = self.w_User[
                        np.array(train_vec[:, 0], dtype="int32"), :
                    ].shape[0]
                    w_big = torch.cat(
                        (
                            self.w_User[
                                np.array(train_vec[:, 0], dtype="int32"), :
                            ],
                            self.w_Item[
                                np.array(train_vec[:, 1], dtype="int32"), :
                            ],
                        ),
                    )
                    obj = self.loss(w_big, ratings, concat_shape, False).item()
                    obj += (
                        0.5
                        * self._lambda
                        * (
                            torch.norm(self.w_User) ** 2
                            + torch.norm(self.w_Item) ** 2
                        )
                    )

                    self.rmse_train.append((obj / pairs_train) ** 0.5)

                # Compute validation error
                if batch == self.num_batches - 1:
                    ratings = test_vec[:, 2]
                    concat_shape = self.w_User[
                        np.array(test_vec[:, 0], dtype="int32"), :
                    ].shape[0]
                    w_big = torch.cat(
                        (
                            self.w_User[
                                np.array(test_vec[:, 0], dtype="int32"), :
                            ],
                            self.w_Item[
                                np.array(test_vec[:, 1], dtype="int32"), :
                            ],
                        ),
                    )

                    loss = self.loss(
                        w_big,
                        ratings,
                        concat_shape,
                        use_regularize=False,
                    ).item()
                    self.rmse_test.append((loss / pairs_test) ** 0.5)

                    # Print info
                    if batch == self.num_batches - 1:
                        print(
                            "Epoch: %d, Training RMSE: %f, Test RMSE %f"
                            % (
                                self.epoch,
                                self.rmse_train[-1],
                                self.rmse_test[-1],
                            ),
                        )

    def predict(self, invID):
        return (
            self.w_Item @ self.w_User[int(invID), :]
        ) + self.mean_inv  # numpy.dot 点乘

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv).numpy())[
                    -k:
                ]  # numpy.argsort索引排序

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = (
                    intersection_cnt.get(test_vec[i, 0], 0) + 1
                )
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype="int32"))

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(
                invPairs_cnt[int(inv)],
            )

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)
