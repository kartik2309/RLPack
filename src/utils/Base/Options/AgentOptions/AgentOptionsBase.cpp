//
// Created by Kartik Rajeshwaran on 2022-07-08.
//

#include "AgentOptionsBase.h"

namespace agent {

    AgentOptionsBase::AgentOptionsBase() = default;

    AgentOptionsBase::AgentOptionsBase(std::shared_ptr<optimizer::OptimizerBase> &optimizer,
                                       std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> &lrScheduler,
                                       float_t gamma, float_t epsilon, float_t minEpsilon,
                                       float_t epsilonDecayRate, int32_t epsilonDecayFrequency,
                                       int32_t modelBackupFrequency, float_t minLr, int32_t batchSize,
                                       int32_t numActions, std::string &savePath, int32_t applyNorm,
                                       int32_t applyNormTo, float_t epsForNorm, int32_t pForNorm,
                                       int32_t dimForNorm) {

        optimizer_ = optimizer;
        lrScheduler_ = lrScheduler;

        gamma_ = gamma;
        epsilon_ = epsilon;
        minEpsilon_ = minEpsilon;
        epsilonDecayRate_ = epsilonDecayRate;
        epsilonDecayFrequency_ = epsilonDecayFrequency;
        assert(targetModelUpdateRate > policyModelUpdateRate);

        modelBackupFrequency_ = modelBackupFrequency;
        minLr_ = minLr;
        batchSize_ = batchSize;
        numActions_ = numActions;

        savePath_ = savePath;

        if (applyNormTo < -1 || applyNormTo > 4) {
            throw std::range_error("Invalid applyNormTo passed!");
        }
        applyNormTo_ = applyNormTo;
        normalization_ = std::make_shared<Normalization>(applyNorm);

        epsForNorm_ = epsForNorm;
        pForNorm_ = pForNorm;
        dimForNorm_ = dimForNorm;
    }

    void AgentOptionsBase::set_optimizer(std::shared_ptr<optimizer::OptimizerBase> &optimizer) {
        optimizer_ = optimizer;
    }

    void AgentOptionsBase::set_lr_scheduler(std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> &lrScheduler) {
        lrScheduler_ = lrScheduler;
    }

    void AgentOptionsBase::gamma(float_t gamma) {
        gamma_ = gamma;
    }

    void AgentOptionsBase::epsilon(float_t epsilon) {
        epsilon_ = epsilon;
    }

    void AgentOptionsBase::min_epsilon(float_t minEpsilon) {
        minEpsilon_ = minEpsilon;
    }

    void AgentOptionsBase::epsilon_decay_rate(float_t epsilonDecayRate) {
        epsilonDecayRate_ = epsilonDecayRate;
    }

    void AgentOptionsBase::epsilon_decay_frequency(int32_t epsilonDecayFrequency) {
        epsilonDecayFrequency_ = epsilonDecayFrequency;
    }

    void AgentOptionsBase::model_backup_frequency(int32_t modelBackFrequency) {
        modelBackupFrequency_ = modelBackFrequency;
    }

    void AgentOptionsBase::min_lr(float_t minLr) {
        minLr_ = minLr;
    }

    void AgentOptionsBase::batch_size(int32_t batchSize) {
        batchSize_ = batchSize;
    }

    void AgentOptionsBase::num_actions(int32_t numActions) {
        numActions_ = numActions;
    }

    void AgentOptionsBase::d_type(torch::ScalarType dType) {
        dType_ = dType;
    }

    void AgentOptionsBase::save_path(std::string &savePath) {
        savePath_ = savePath;
    }

    void AgentOptionsBase::apply_norm(int32_t applyNorm) {
        normalization_ = std::make_shared<Normalization>(applyNorm);
        applyNorm_ = applyNorm;
    }

    void AgentOptionsBase::apply_norm_to(int32_t applyNormTo) {
        if (applyNormTo < -1 || applyNormTo > 4) {
            throw std::range_error("Invalid applyNormTo passed!");
        }
        applyNormTo_ = applyNormTo;
    }

    void AgentOptionsBase::eps_for_norm(float_t epsForNorm) {
        epsForNorm_ = epsForNorm;
    }

    void AgentOptionsBase::p_for_norm(int32_t pForNorm) {
        pForNorm_ = pForNorm;
    }

    void AgentOptionsBase::dim_for_norm(int32_t dimForNorm) {
        dimForNorm_ = dimForNorm;
    }

    std::shared_ptr<optimizer::OptimizerBase> AgentOptionsBase::get_optimizer() {
        return optimizer_;
    }

    std::shared_ptr<optimizer::lrScheduler::LrSchedulerBase> AgentOptionsBase::get_lr_scheduler() {
        return lrScheduler_;
    }

    float_t AgentOptionsBase::get_gamma() const {
        return gamma_;
    }

    float_t AgentOptionsBase::get_epsilon() const {
        return epsilon_;
    }

    float_t AgentOptionsBase::get_min_epsilon() const {
        return minEpsilon_;
    }

    float_t AgentOptionsBase::get_epsilon_decay_rate() const {
        return epsilonDecayRate_;
    }

    int32_t AgentOptionsBase::get_epsilon_decay_frequency() const {
        return epsilonDecayFrequency_;
    }

    int32_t AgentOptionsBase::get_model_backup_frequency() const {
        return modelBackupFrequency_;
    }

    float_t AgentOptionsBase::get_min_lr() const {
        return minLr_;
    }

    int32_t AgentOptionsBase::get_batch_size() const {
        return batchSize_;
    }

    int32_t AgentOptionsBase::get_num_actions() const {
        return numActions_;
    }

    torch::ScalarType AgentOptionsBase::get_d_type() const {
        return dType_;
    }

    std::string AgentOptionsBase::get_save_path() {
        return savePath_;
    }

    int32_t AgentOptionsBase::get_apply_norm() const {
        return applyNorm_;
    }

    int32_t AgentOptionsBase::get_apply_norm_to() const {
        return applyNormTo_;
    }

    float_t AgentOptionsBase::get_eps_for_norm() const {
        return epsForNorm_;
    }

    int32_t AgentOptionsBase::get_p_for_norm() const {
        return pForNorm_;
    }

    int32_t AgentOptionsBase::get_dim_for_norm() const {
        return dimForNorm_;
    }

    std::shared_ptr<Normalization> AgentOptionsBase::get_normalizer() {
        return normalization_;
    }

    AgentOptionsBase::~AgentOptionsBase() = default;

}
