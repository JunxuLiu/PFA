import copy
import numpy as np
import torch
from typing import List

from myopacus import PrivacyEngine
from strategies.fed_avg_base import FederatedLearning
from strategies.strategies_utils import _Model, evaluate_model_on_tests
from modules.lanczos import Lanczos

class PFA(FederatedLearning):
    def __init__(
        self,
        **kwargs
    ):
        super(FederatedLearning, self).__init__(**kwargs)

        if self.privacy_engine is not None:
            self.public_clients = self._set_public_clients()

    def _set_public_clients(self, percent = 0.1):
        sorted_eps = np.sort(self.privacy_engine.target_epsilons)
        threshold = sorted_eps[-int(percent * self.num_clients)]
        self.public_clients = list(np.where(np.array(self.privacy_engine.target_epsilons) >= threshold)[0])

    def _client_sampling(self):
        # For PFA, we require the subset contains at least one public and one private client.
        selected_clients = []
        check = 50
        while check and len(set(selected_clients).intersection(set(self.public_clients))) == 0:
            check -= 1
            mask = (np.random.random(self.num_clients) < self.client_rate)
            selected_clients = np.where(mask == True)[0]
        print("Index of the selected clients: ", selected_clients)
        return selected_clients

    def _standardize(self, M):
        '''Compute the mean of every dimension of the whole dataset'''
        [n, m] = M.shape
        if m == 1:
            print(m==1)
            return M, np.zeros(n)
        # calculate the mean 
        mean = np.dot(M,np.ones((m,1), dtype=np.float32)) / m
        return M - mean, mean.flatten()


    def _eigen_by_lanczos(self, mat):
        T, V = Lanczos(mat, self.lanczos_iter)
        T_evals, T_evecs = np.linalg.eig(T)
        idx = T_evals.argsort()[-1 : -(self.proj_dims+1) : -1]
        Vk = np.dot(V.T, T_evecs[:,idx])
        return Vk


    def _global_aggregation(self, local_updates):
        """Carry out the global parameter aggregation step."""
        pub_updates, priv_updates = local_updates["public"], local_updates["private"]
        num_weights = len(pub_updates[0]["updates"])
        shape_weights = [var.shape for var in pub_updates[0]["updates"]]
        aggregated_delta_weights = [ None ] * num_weights

        if len(pub_updates) and len(priv_updates):
            mean_priv_updates = [np.mean(priv_updates[i], 0) for i in range(num_weights)]
            mean_pub_updates = [np.mean(pub_updates[i], 0) for i in range(num_weights)]
            mean_proj_priv_updates = [0] * num_weights
            aggregated_delta_weights = [0] * num_weights
            
            for i in range(num_weights):
                _pub_updates, mean = self._standardize(pub_updates[i].T)
                Vk = self._eigen_by_lanczos(_pub_updates.T)
                mean_proj_priv_updates[i] = np.dot(Vk, np.dot(Vk.T, (mean_priv_updates[i] - mean))) + mean
                aggregated_delta_weights[i] = ((len(priv_updates) * mean_proj_priv_updates[i] + len(pub_updates) * mean_pub_updates[i]) /
                                  len(priv_updates)+len(pub_updates)).reshape(shape_weights[i])

        elif len(local_updates["public"]) and not len(priv_updates):
            aggregated_delta_weights = [np.mean(pub_updates[i], 0).reshape(shape_weights[i]) for i in range(num_weights)]

        else:
            raise ValueError('Cannot process the projection without private local updates.')

        # for idx_weight in range(len(local_updates[0]["updates"])):
        #     aggregated_delta_weights[idx_weight] = sum(
        #         [
        #             local_updates[idx_client]["updates"][idx_weight]
        #             * local_updates[idx_client]["n_samples"]
        #             for idx_client in range(len(local_updates))
        #         ]
        #     )
        #     aggregated_delta_weights[idx_weight] /= float(total_samples)

        return aggregated_delta_weights
    
    def perform_round(self):
        # Sampling clients
        selected_clients_idx = self._client_sampling()

        # pub_updates, priv_updates = list(), list()
        local_updates = {
            "public": list(),
            "private": list()
        }
        for _model in np.array(self.models_list)[selected_clients_idx]:
            flag = ("public" if _model.id in self.public_clients else "private")
            # Local Optimization
            _local_previous_state = _model._get_current_params()
            self._local_optimization(_model)
            _local_next_state = _model._get_current_params()

            # Recovering updates
            updates = [
                new - old for new, old in zip(_local_next_state, _local_previous_state)
            ]
            del _local_next_state

            # Reset local model
            for p_new, p_old in zip(_model.model.parameters(), _local_previous_state):
                p_new.data = torch.from_numpy(p_old).to(p_new.device)
            del _local_previous_state
            local_updates[flag].append({"updates": updates, "n_samples": len(_model._train_dl.dataset)})

        # Projected Federated Averaging
        aggregated_delta_weights = self._global_aggregation(local_updates)

        # Update models
        for _model in self.models_list:
            _model._update_params(aggregated_delta_weights)