//
// Created by Kartik Rajeshwaran on 2022-06-27.
//

#ifndef RLPACK_SRC_DQN_AGENT_TPP_
#define RLPACK_SRC_DQN_AGENT_TPP_

#include "Agent.h"

namespace dqn {

    template<class ModelClass, class Optimizer>
    Agent<ModelClass, Optimizer>::Agent(
            ModelClass &targetModel, ModelClass &policyModel, Optimizer &optimizer,
            float_t gamma, float_t epsilon, float_t minEpsilon, float_t epsilonDecayRate,
            int32_t epsilonDecayFrequency, int32_t memoryBufferSize, int32_t targetModelUpdateRate,
            int32_t policyModelUpdateRate, int32_t batchSize, int32_t numActions,
            std::string &savePath, float_t tau, int32_t applyNorm, int32_t applyNormTo,
            float_t epsForNorm, int32_t pForNorm, int32_t dimForNorm
    ) {
        targetModel_ = targetModel;
        policyModel_ = policyModel;
        optimizer_ = optimizer;

        gamma_ = gamma;
        epsilon_ = epsilon;
        minEpsilon_ = minEpsilon;
        epsilonDecayRate_ = epsilonDecayRate;
        epsilonDecayFrequency_ = epsilonDecayFrequency;
        memoryBufferSize_ = memoryBufferSize;
        assert(targetModelUpdateRate > policyModelUpdateRate);

        targetModelUpdateRate_ = targetModelUpdateRate;
        policyModelUpdateRate_ = policyModelUpdateRate;
        batchSize_ = batchSize;
        numActions_ = numActions;

        savePath_ = savePath;

        if (tau < 0 || tau > 1){
            throw std::range_error("Invalid value for tau passed! Expected value is between 0 and 1");
        }
        tau_ = tau;

        if (applyNormTo < -1 || applyNormTo > 2) {
            throw std::range_error("Invalid applyNormTo passed!");
        }
        applyNormTo_ = applyNormTo;
        normalization_ = new Normalization(applyNorm);

        epsForNorm_ = epsForNorm;
        pForNorm_ = pForNorm;
        dimForNorm_ = dimForNorm;

        memoryBuffer.reserve(memoryBufferSize);
    }

    template<class ModelClass, class Optimizer>
    int32_t Agent<ModelClass, Optimizer>::train(torch::Tensor &stateCurrent, torch::Tensor &stateNext, float reward,
                                                int action, int done) {

        memoryBuffer.push_back(stateCurrent, stateNext, reward, action, done);

        if ((policyModelUpdateCounter % policyModelUpdateRate_ == 0) && (memoryBuffer.size() >= batchSize_)) {
            train_policy_model();
        }

        if (targetModelUpdateCounter % targetModelUpdateRate_ == 0) {
            update_target_model();
        }

        if (memoryBuffer.size() == memoryBufferSize_) {
            clear_memory();
        }

        if (done == 1) {
            if (epsilonDecayCounter % epsilonDecayFrequency_ == 0){
                decay_epsilon();
            }
            epsilonDecayCounter += 1;
        }

        policyModelUpdateCounter += 1;
        targetModelUpdateCounter += 1;

        stateCurrent = stateCurrent.unsqueeze(0);
        action = policy(stateCurrent);
        return action;
    }

    template<class ModelClass, class Optimizer>
    void Agent<ModelClass, Optimizer>::train_policy_model() {
        Memory *randomExperiences = load_random_experiences();

        torch::Tensor statesCurrent = randomExperiences->stack_current_states();
        torch::Tensor statesNext = randomExperiences->stack_next_states();
        torch::Tensor rewards = randomExperiences->stack_rewards();
        torch::Tensor actions = randomExperiences->stack_actions();
        torch::Tensor dones = randomExperiences->stack_dones();

        delete randomExperiences;

        if (applyNormTo_ >= 0) {
            statesCurrent = normalization_->apply_normalization(statesCurrent, epsForNorm_, pForNorm_, dimForNorm_);
            statesNext = normalization_->apply_normalization(statesNext, epsForNorm_, pForNorm_, dimForNorm_);
        }

        if (applyNormTo_ == 1) {
            rewards = normalization_->apply_normalization(rewards, epsForNorm_, pForNorm_, dimForNorm_);
        }

        policyModel_->train();
        torch::Tensor tdValue;
        {
            targetModel_->eval();
            torch::NoGradGuard guard;
            torch::Tensor qValuesTarget = targetModel_->forward(statesNext);
            tdValue = temporal_difference(rewards, qValuesTarget, dones);

            if (applyNormTo_ == 2) {
                tdValue = normalization_->apply_normalization(tdValue, epsForNorm_, pForNorm_, dimForNorm_);
            }
        }

        torch::Tensor qValuesPolicy = policyModel_->forward(statesCurrent);
        torch::Tensor qValuesPolicyGathered = qValuesPolicy.gather(-1, actions);

        if (qValuesPolicyGathered.isnan().any().item<bool>() || tdValue.isnan().any().item<bool>()) {
            BOOST_LOG_TRIVIAL(warning) << "NaN values encountered during training! "
                                          "This may lead to instability in models" << std::endl;
        }

        optimizer_->zero_grad();
        torch::Tensor loss = huberLoss_(tdValue.detach(), qValuesPolicyGathered);
        loss.backward();
        optimizer_->step();
    }

    template<class ModelClass, class Optimizer>
    void Agent<ModelClass, Optimizer>::update_target_model() {
        {

            torch::NoGradGuard noGradGuard;
            std::vector<torch::Tensor> policyParameters = policyModel_->parameters();
            std::vector<torch::Tensor> targetParameters = targetModel_->parameters();

            if (policyParameters.size() != targetParameters.size()) {
                throw std::length_error("Target and policy model have different number of parameters!");
            }

            uint64_t parametersLength = policyParameters.size();

            for (uint64_t idx = 0; idx != parametersLength; idx++) {
                targetParameters.at(idx).copy_(
                        tau_ * policyParameters.at(idx) + (1 - tau_) * targetParameters.at(idx),
                        true
                );
            }

            std::vector<torch::Tensor> policyParametersForCheck = policyModel_->parameters();
            std::vector<torch::Tensor> targetParametersForCheck = targetModel_->parameters();

        }
    }

    template<class ModelClass, class Optimizer>
    Memory *Agent<ModelClass, Optimizer>::load_random_experiences() {
        std::vector<int32_t> loadedIndices(policyModelUpdateRate_);
        boost::random::random_device rd;

        boost::mt19937 generator(rd);
        boost::uniform_int<int32_t> randomIndices(0, (int32_t) memoryBuffer.size());
        boost::variate_generator<boost::mt19937, boost::uniform_int<int32_t>> variateGenerator(generator,
                                                                                               randomIndices);

        boost::push_back(loadedIndices, boost::irange(0, (int32_t) memoryBuffer.size(), 1));
        boost::range::random_shuffle(loadedIndices, variateGenerator);

        auto *loadedExperiences = new Memory();

        for (int32_t index: loadedIndices) {
            memoryBuffer.at(loadedExperiences, index);

            if (loadedExperiences->size() == batchSize_) {
                break;
            }
        }

        if (loadedExperiences->size() != batchSize_) {
            throw std::runtime_error("Loaded Experience Sizes do not match Batch Size!");
        }

        loadedIndices.clear();

        return loadedExperiences;
    }

    template<class ModelClass, class Optimizer>
    torch::Tensor Agent<ModelClass, Optimizer>::temporal_difference(torch::Tensor &rewards, torch::Tensor &qValues,
                                                                    torch::Tensor &dones) {
        std::tuple<torch::Tensor, torch::Tensor> qValuesTuple = qValues.max(-1, true);
        torch::Tensor qValuesMax = std::get<0>(qValuesTuple);
        torch::Tensor tdValue = rewards + ((gamma_ * qValuesMax) * (1 - dones));

        return tdValue;
    }

    template<class ModelClass, class Optimizer>
    int32_t Agent<ModelClass, Optimizer>::policy(torch::Tensor &stateCurrent) {
        int32_t action;
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<float> distributionP(0, 1);
        std::uniform_int_distribution<int> distributionAction(0, numActions_ - 1);
        float_t p = distributionP(generator);

        if (p < epsilon_) {
            action = distributionAction(generator);
        } else {
            {
                policyModel_->eval();
                torch::NoGradGuard guard;

                if (applyNormTo_ >= 0) {
                    stateCurrent = normalization_->apply_normalization(stateCurrent);
                }

                torch::Tensor qValues = policyModel_->forward(stateCurrent);
                torch::Tensor actionTensor = qValues.argmax(-1);

                action = actionTensor.item<int32_t>();
            }
        }
        return action;
    }

    template<class ModelClass, class Optimizer>
    void Agent<ModelClass, Optimizer>::decay_epsilon() {
        if (epsilon_ > minEpsilon_) {
            epsilon_ *= epsilonDecayRate_;
        }

        if (epsilon_ < minEpsilon_) {
            epsilon_ = minEpsilon_;
        }
    }

    template<class ModelClass, class Optimizer>
    void Agent<ModelClass, Optimizer>::clear_memory() {
        memoryBuffer.clear();
    }

    template<class ModelClass, class Optimizer>
    void Agent<ModelClass, Optimizer>::save() {

        std::string policyModelPath = savePath_;
        std::string targetModelPath = savePath_;
        std::string statePath = savePath_;

        torch::serialize::OutputArchive stateArchive;
        torch::TensorOptions optionsForEpsilonAsTensor = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor epsilonAsTensor = torch::full({}, epsilon_, optionsForEpsilonAsTensor);
        stateArchive.write("epsilon", epsilonAsTensor);

        torch::save(policyModel_, policyModelPath.append("_policy.pt"));
        torch::save(targetModel_, targetModelPath.append("_target.pt"));
        stateArchive.save_to(statePath.append("stateValues.pt"));
    }

    template<class ModelClass, class Optimizer>
    void Agent<ModelClass, Optimizer>::load() {
        std::string policyModelPath = savePath_;
        std::string targetModelPath = savePath_;
        std::string statePath = savePath_;

        torch::serialize::InputArchive inputArchive;
        torch::TensorOptions optionsForEpsilonAsTensor = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor epsilonAsTensor = torch::full({}, 0, optionsForEpsilonAsTensor);
        inputArchive.load_from(statePath);
        inputArchive.read("epsilon", epsilonAsTensor);
        epsilon_ = epsilonAsTensor.template item<float_t>();

        torch::load(policyModel_, policyModelPath.append("_policy.pt"));
        torch::load(targetModel_, targetModelPath.append("_target.pt"));

    }

    template<class ModelClass, class Optimizer>
    Agent<ModelClass, Optimizer>::~Agent() = default;

}

#endif//RLPACK_SRC_DQN_AGENT_TPP_
