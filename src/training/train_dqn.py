import torch
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.multiagent.marl import MultiAgentOffPolicyAlgorithm
from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory

import config
from training.base_trainer import BasePokerTrainer
from evaluation.evaluator_dqn import DQNEvaluator
from models import MaskedActor
from opponents import FrozenDQN, PassiveAlgorithm, SeededMixedAlgorithm
from phases import DynamicOpponentAlgorithm
from paths import DQN_CHECKPOINT_DIR

def phase_one_epsilon(env_step: int) -> float:
    """Liniowo zmniejsz eksplorację z 1.0 do 0.1 przez 720 tys. akcji.

    Po osiągnięciu minimum agent nadal losuje 10% decyzji. Zapobiega to zbyt
    wczesnemu przywiązaniu do strategii poznanej głównie na pasywnych botach.
    """
    progress = min(max(env_step, 0) / config.DQN_PHASE1_EPS_DECAY_STEPS, 1.0)
    return max(
        config.DQN_RAND_PHASE_EPS_MIN,
        config.DQN_EPS_MAX
        + progress * (config.DQN_RAND_PHASE_EPS_MIN - config.DQN_EPS_MAX),
    )


def configure_macbook_cpu_runtime() -> None:
    """Ustaw PyTorch zgodnie z benchmarkiem wykonanym na MacBooku Air M2.

    Osiem osobnych procesów zbiera doświadczenia ze środowisk pokerowych.
    Aktualizacja małej sieci DQN jest natomiast najszybsza na jednym wątku
    CPU; MPS i wielowątkowy PyTorch dodawały więcej narzutu niż pracy.
    """
    torch.set_num_threads(config.TORCH_NUM_THREADS)
    torch.set_num_interop_threads(config.TORCH_NUM_INTEROP_THREADS)


def ensure_finite_model(model: torch.nn.Module, env_step: int) -> None:
    """Przerwij trening od razu, gdy wagi zawierają NaN albo nieskończoność."""
    invalid_parameters = [
        name
        for name, parameter in model.named_parameters()
        if not torch.isfinite(parameter).all()
    ]
    if invalid_parameters:
        names = ", ".join(invalid_parameters)
        raise FloatingPointError(
            f"Niestabilny DQN po {env_step:,} akcjach; "
            f"niepoprawne parametry: {names}"
        )


def validate_phase_one_configuration() -> None:
    """Wykryj literówki w konfiguracji przed kosztownym uruchomieniem."""
    if config.TRAINING_PHASE != 1:
        raise ValueError("Ten skrypt fazy rozgrzewkowej wymaga TRAINING_PHASE = 1")
    if not np.isclose(sum(config.PHASE1_OPPONENT_WEIGHTS.values()), 1.0):
        raise ValueError("Wagi przeciwników fazy 1 muszą sumować się do 1.0")
    if config.DQN_PHASE1_EPS_DECAY_STEPS > (
        config.DQN_MAX_EPOCHS * config.DQN_STEPS_PER_EPOCH
    ):
        raise ValueError("Epsilon nie zdąży osiągnąć minimum przed końcem fazy 1")


class DQNPokerTrainer(BasePokerTrainer):
    def setup_and_train(self):
        # Konfiguracja Ucznia
        net_learner = MaskedActor(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)

        policy_learner = DiscreteQLearningPolicy(
            model=net_learner,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            eps_training=config.DQN_EPS_MAX,
            eps_inference=0.0
        )

        if self.training_phase in ["SELF", "ADVANCED"]:
            try:
                #TODO: trzeba zrobić jakiś lepszy system wczytywania modelu do ucznia
                policy_learner.load_state_dict(torch.load(DQN_CHECKPOINT_DIR / 'final.pth', map_location=self.device, weights_only=True))
                print("Wczytano wagi ucznia z poprzedniej fazy!")
            except FileNotFoundError:
                print("Brak końcowego modelu DQN dla ucznia, start od zera.")
        
        dqn_learner = DQN(
            policy=policy_learner,
            optim=AdamOptimizerFactory(lr=config.DQN_LEARNING_RATE),
            gamma=config.DQN_GAMMA, # TODO: zobaczyć czy lepiej nie ustawić 0.95
            n_step_return_horizon=3,
            target_update_freq=config.DQN_TARGET_NET_UPDATE
        )

        # Konfiguracja Przeciwników
        net_opponent = MaskedActor(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)
        policy_opponent = DiscreteQLearningPolicy(
            model=net_opponent,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            eps_training=config.DQN_OPONENT_EPS,  # mała losowość, żeby nie był bardzo przewidywalny
            eps_inference=0.0
        )

        policy_opponent.eval()

        for param in policy_opponent.parameters():
            param.requires_grad = False

        frozen_opponent = FrozenDQN(
            policy=policy_opponent,
            optim=AdamOptimizerFactory(lr=0.0),
            gamma=config.DQN_GAMMA,
            n_step_return_horizon=3,
            target_update_freq=0
        )

        # W fazie 1 zamrożony model nie jest przeciwnikiem, dlatego nie
        # wczytujemy starego checkpointu. Dawny model 68-wejściowy nie pasuje
        # do obecnej sieci z 222 obserwacjami.
        if self.phase_name in {"self", "advanced"}:
            try:
                frozen_opponent.policy.load_state_dict(
                    torch.load(
                        DQN_CHECKPOINT_DIR / "best.pth",
                        map_location=self.device,
                        weights_only=True,
                    )
                )
            except FileNotFoundError:
                pass

        # Inne agenty
        random_agent = MARLRandomDiscreteMaskedOffPolicyAlgorithm(action_space=self.env.action_space)
        passive_agent = PassiveAlgorithm(action_space=self.env.action_space)
        mixed_agent = SeededMixedAlgorithm(action_space=self.env.action_space, seed=12345)


        # Złożenie środowiska MARL
        if self.training_phase == 1:
            available_opponents = {
                "random": random_agent.policy,
                "passive": passive_agent.policy,
                "mixed": mixed_agent.policy
            }

            # Jedno źródło konfiguracji gwarantuje, że trening i ewaluacja
            # używają tej samej mieszanki 50% Random / 40% Passive / 10% Mixed.
            opponent_weights = config.PHASE1_OPPONENT_WEIGHTS

            opponent_1 = DynamicOpponentAlgorithm(self.env.action_space, available_opponents, opponent_weights, seed=10_001)
            opponent_2 = DynamicOpponentAlgorithm(self.env.action_space, available_opponents, opponent_weights, seed=10_002)
            opponent_3 = DynamicOpponentAlgorithm(self.env.action_space, available_opponents, opponent_weights, seed=10_003)

            agents = [dqn_learner, opponent_1, opponent_2, opponent_3]
        # TODO: trzeba zrobić inne fazy treningu
        else:
            raise NotImplementedError(f"Nieobsługiwana faza DQN: {self.training_phase}")
            
        marl_algo = MultiAgentOffPolicyAlgorithm(algorithms=agents, env=self.env)

        # Kolektory
        buffer = VectorReplayBuffer(config.DQN_BUFFER_SIZE, len(self.train_envs))
        train_collector = Collector(
            marl_algo, 
            self.train_envs, 
            buffer, 
            exploration_noise=True,
        )

        test_collector = Collector(
            marl_algo, 
            self.test_envs, 
            exploration_noise=False,
        )

        print("Zapełnianie bufora pierwszymi losowymi danymi...")
        train_collector.collect(n_step=config.DQN_BUFFER_WARMUP, random=True, reset_before_collect=True)

        # Funkcje trenujące z logiką DQN (Epsilon Decay)
        # eps definiuje jak często podejmowane są losowe decyzje
        # TODO: można tu coś pokombinować, ale raczej jest git
        def train_fn(epoch, env_step):
            if self.phase_name in {"1", "random"}:
                eps = phase_one_epsilon(env_step)
            else:
                eps = max(config.DQN_OTHER_PHASE_EPS_MIN, config.DQN_OTHER_PHASE_EPS_MAX - env_step / (config.DQN_OTHER_PHASE_EPS_DECAY * self.total_steps))
            dqn_learner.policy.set_eps_training(eps)

            # Callback jest wykonywany na granicy epok. Kontrola po poprzedniej
            # serii aktualizacji zatrzyma proces, zanim NaN uszkodzi kolejne
            # checkpointy albo cały replay buffer.
            ensure_finite_model(net_learner, env_step)

            self.run_periodic_evaluation(
                env_step=env_step,
                learner_policy=dqn_learner.policy,
                opponent_policy=policy_opponent,
            )

        def test_fn(epoch, env_step):
            dqn_learner.policy.set_eps_inference(0.0)

        # Trener
        trainer_params = OffPolicyTrainerParams(
            max_epochs=self.max_epochs,
            epoch_num_steps=self.steps_per_epoch,
            collection_step_num_env_steps=config.DQN_COLLECTION_STEPS,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=config.EVAL_SMOKE_TOURNAMENTS,
            batch_size=config.DQN_BATCH_SIZE,
            training_fn=train_fn,
            test_fn=test_fn,
            multi_agent_return_reduction=lambda ret: ret[:, 0]
        )

        # Pomiar przed treningiem daje uczciwy punkt odniesienia dla wszystkich
        # późniejszych checkpointów tej samej, losowo zainicjalizowanej sieci.
        self.run_initial_evaluation(learner_policy=dqn_learner.policy)
        print("Rozpoczęcie treningu DQN...")
        result = OffPolicyTrainer(algorithm=marl_algo, params=trainer_params).run()
        print(f"\n=== Trening Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        # Callback treningowy działa przed kolekcją, dlatego ostatni próg
        # miliona akcji obsługujemy jawnie po zwróceniu końcowych statystyk.
        # Zapewnia to checkpoint i walidację dokładnie dla kroku 1 000 000.
        ensure_finite_model(net_learner, result.train_step)
        self.run_periodic_evaluation(
            env_step=result.train_step,
            learner_policy=dqn_learner.policy,
            opponent_policy=policy_opponent,
        )
        self.run_final_evaluation(
            learner_policy=dqn_learner.policy,
            env_step=result.train_step,
        )

if __name__ == "__main__":
    validate_phase_one_configuration()
    configure_macbook_cpu_runtime()
    print(
        "Konfiguracja fazy 1: "
        f"{config.DQN_MAX_EPOCHS * config.DQN_STEPS_PER_EPOCH:,} akcji treningowych, "
        f"warm-up {config.DQN_BUFFER_WARMUP:,}, "
        f"{config.DQN_NUM_TRAIN_ENVS} środowisk, "
        f"{config.TORCH_NUM_THREADS} wątek PyTorch."
    )
    trainer = DQNPokerTrainer(
        algo_name="dqn",
        training_phase=config.TRAINING_PHASE,
        evaluator_class=DQNEvaluator,
        num_train_envs=config.DQN_NUM_TRAIN_ENVS,
        num_test_envs=config.DQN_NUM_TEST_ENVS,
        max_epochs=config.DQN_MAX_EPOCHS,
        steps_per_epoch=config.DQN_STEPS_PER_EPOCH
    )
    trainer.setup_and_train()
