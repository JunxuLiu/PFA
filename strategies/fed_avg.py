import numpy as np
import time
import torch
from typing import List

from myopacus import PrivacyEngine
from strategies.strategies_utils import _Model, evaluate_model_on_tests

def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

class FedAvg:
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
        test_dataloaders: List, # added by Junxu
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
        bits_counting_function: callable = None,
        logdir: str = "./runs",
        log_basename: str = "fed_avg",
        seed: int = None,
        **kwargs
    ):
        self.privacy_engine = privacy_engine
        
        self.client_rate = client_rate
        self.num_rounds = num_rounds
        self.num_steps = num_steps

        self.log = log
        self.log_period = log_period
        self.log_basename = log_basename
        self.logdir = logdir
        self._seed = seed
        set_random_seed(self._seed)

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

        if self.privacy_engine is not None:
            assert (self.privacy_engine.accountant.mechanism() == "idp"), \
                 "DataType of `privacy_engine.accountant` must be `IndividualAccountant` in FL setup."
            
            for _model in self.models_list:
                _model._make_private(self.privacy_engine)

        self.num_clients = len(training_dataloaders)
        self.bits_counting_function = bits_counting_function

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

    def perform_round(self):
        """Does a single federated averaging round. The following steps will be
        performed:

        - each model will be trained locally for num_steps batches.
        - the parameter updates will be collected and averaged. Averages will be
          weighted by the number of samples in each client
        - the averaged updates willl be used to update the local model
        """
        local_updates = list()
        total_number_of_samples = 0
        selected_idx_client = []
        while len(selected_idx_client) == 0:
            mask = (np.random.random(self.num_clients) < self.client_rate)
            selected_idx_client = np.where(mask == True)[0]
            print("selected_idx_client: ", selected_idx_client)
        
        for _model in np.array(self.models_list)[selected_idx_client]:
            print(f"Client {_model.client_id} ...")
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

            if self.bits_counting_function is not None:
                self.bits_counting_function(updates)
            
            local_updates.append({"updates": updates, "n_samples": len(_model._train_dl.dataset)})
            total_number_of_samples += len(_model._train_dl.dataset)

        # Aggregation step
        aggregated_delta_weights = [
            None for _ in range(len(local_updates[0]["updates"]))
        ]
        for idx_weight in range(len(local_updates[0]["updates"])):
            aggregated_delta_weights[idx_weight] = sum(
                [
                    local_updates[idx_client]["updates"][idx_weight]
                    * local_updates[idx_client]["n_samples"]
                    for idx_client in range(len(selected_idx_client))
                ]
            )
            aggregated_delta_weights[idx_weight] /= float(total_number_of_samples)

        # Update models
        for _model in self.models_list:
            _model._update_params(aggregated_delta_weights)

    # def run(self, metric, device):
    def run(self):
        """This method performs self.nrounds rounds of averaging
        and returns the list of models.
        """
        all_round_results = []
        for r in range(self.num_rounds):
            self.perform_round()
            perf, y_true_dict, y_pred_dict = evaluate_model_on_tests(self.models_list, return_pred=True)
            if self.privacy_engine:
                ret = self.privacy_engine.accountant.get_epsilon(delta=self.privacy_engine.target_delta, mode="max")
                print("current privacy cost of all clients: ", ret)

            correct = np.array(
                [v for _, v in perf.items()]
            ).sum()
            total = np.array(
                [len(v) for _, v in y_true_dict.items()]
            ).sum()
            print(f"Round={r}, perf={list(perf.values())}, mean perf={correct}/{total} ({correct/total:.4f}%)")
            all_round_results.append(round(correct/total, 4))

        return [m.model for m in self.models_list], all_round_results
