import abc
import numpy as np
import torch
from typing import List

from strategies.strategies_utils import _Model, evaluate_model_on_tests
from myopacus import PrivacyEngine

class FederatedLearning(metaclass=abc.ABCMeta):
    """Federated Averaging Strategy class.

    The Federated Averaging strategy is the most simple centralized FL strategy.
    Each client first trains his version of a global model locally on its data,
    the states of the model of each client are then weighted-averaged and returned
    to each client for further training.

    References
    ----------
    - https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    training_dataloaders : List
        The list of training dataloaders from multiple training centers.
    model : torch.nn.Module
        An initialized torch model.
    loss : torch.nn.modules.loss._Loss
        The loss to minimize between the predictions of the model and the
        ground truth.
    optimizer_class : torch.optim.Optimizer
        The class of the torch model optimizer to use at each step.
    learning_rate : float
        The learning rate to be given to the optimizer_class.
    num_steps : int
        The number of steps to do on each client at each round.
    num_rounds : int
        The number of communication rounds to do.
    log: bool, optional
        Whether or not to store logs in tensorboard. Defaults to False.
    log_period: int, optional
        If log is True then log the loss every log_period batch updates.
        Defauts to 100.
    bits_counting_function : Union[callable, None], optional
        A function making sure exchanges respect the rules, this function
        can be obtained by decorating check_exchange_compliance in
        flamby.utils. Should have the signature List[Tensor] -> int.
        Defaults to None.
    logdir: str, optional
        Where logs are stored. Defaults to ./runs.
    log_basename: str, optional
        The basename of the created log_file. Defaults to fed_avg.
    """

    def __init__(
        self,
        training_dataloaders: List, 
        test_dataloaders: List,
        model: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        metric: callable,
        optimizer_class: torch.optim.Optimizer,
        learning_rate: float,
        client_rate: float,
        num_steps: int,
        num_rounds: int,
        privacy_engine: PrivacyEngine = None,
        device: str = "cuda:0",
        log: bool = False,
        log_period: int = 100,
        logdir: str = "./runs",
        log_basename: str = "fed_avg",
        seed: int = None
    ):
        self.client_rate = client_rate
        self.num_rounds = num_rounds
        self.num_steps = num_steps

        self.log = log
        self.log_period = log_period
        self.log_basename = log_basename
        self.logdir = logdir
        self._seed = seed

        self.models_list = [
            _Model(
                model=model,
                optimizer_class=optimizer_class,
                lr=learning_rate,
                train_dl=_train_dl,
                test_dl=_test_dl,
                device=device,
                metric=metric,
                loss=loss,
                log=self.log,
                client_id=i,
                log_period=self.log_period,
                log_basename=self.log_basename,
                logdir=self.logdir,
                seed=self._seed,
            )
            for i, (_train_dl, _test_dl) in enumerate(list(zip(training_dataloaders, test_dataloaders)))
        ]
        self.num_clients = len(training_dataloaders)
        
        if privacy_engine is not None:
            assert (privacy_engine.accountant.mechanism() == "idp"), \
                 "DataType of `privacy_engine.accountant` must be `IndividualAccountant` in FL setup."
            self.privacy_engine = privacy_engine
            for _model in self.models_list:
                _model._make_private(self.privacy_engine)

    def _client_sampling(self):
        selected_clients = []
        check = 50
        while check and len(selected_clients) == 0:
            check -= 1
            mask = (np.random.random(self.num_clients) < self.client_rate)
            selected_clients = np.where(mask == True)[0]
        print("Index of the selected clients: ", selected_clients)
        return selected_clients

    def _local_optimization(self, _model: _Model):
        """Carry out the local optimization step."""
        if self.privacy_engine is None:
            _model._local_train(self.num_steps)
        elif not (self.privacy_engine.accountant.mechanism() == "idp"):
            _model._local_train(self.num_steps, \
                                privacy_accountant=self.privacy_engine.accountant)
        else:
            _model._local_train(self.num_steps, \
                                privacy_accountant=self.privacy_engine.accountant.accountants[_model.client_id])


    def _global_aggregation(self, local_updates):
        """Carry out the global parameter aggregation step."""
        aggregated_delta_weights = [
            None for _ in range(len(local_updates[0]["updates"]))
        ]
        total_samples = sum([
            local_updates[_]["n_samples"] for _ in range(len(local_updates))
        ])
        for idx_weight in range(len(local_updates[0]["updates"])):
            aggregated_delta_weights[idx_weight] = sum(
                [
                    local_updates[idx_client]["updates"][idx_weight]
                    * local_updates[idx_client]["n_samples"]
                    for idx_client in range(len(local_updates))
                ]
            )
            aggregated_delta_weights[idx_weight] /= float(total_samples)

        return aggregated_delta_weights
    
    def perform_round(self):
        """Does a single federated averaging round. The following steps will be
        performed:

        - each model will be trained locally for num_steps batches.
        - the parameter updates will be collected and averaged. 
          Averages will be weighted by the number of samples in each client
        - the averaged updates will be used to update the local model
        """
        # Sampling clients
        selected_clients_idx = self._client_sampling()

        local_updates = list()
        for _model in np.array(self.models_list)[selected_clients_idx]:
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
            
            local_updates.append({"updates": updates, "n_samples": len(_model._train_dl.dataset)})

        # Global params aggregation
        aggregated_delta_weights = self._global_aggregation(local_updates)

        # Update models
        for _model in self.models_list:
            _model._update_params(aggregated_delta_weights)


    def run(self):
        """This method performs self.nrounds rounds of averaging and returns the list of models.
        """
        all_round_results = []
        for r in range(self.num_rounds):
            self.perform_round()
            perf, y_true_dict, y_pred_dict = evaluate_model_on_tests(self.models_list, return_pred=True)

            correct = np.array(
                [v for _, v in perf.items()]
            ).sum()
            total = np.array(
                [len(v) for _, v in y_true_dict.items()]
            ).sum()
            print(f"Round={r}, perf={list(perf.values())}, mean perf={correct}/{total} ({correct/total:.4f}%)")
            all_round_results.append(round(correct/total, 4))

        return [m.model for m in self.models_list], all_round_results
