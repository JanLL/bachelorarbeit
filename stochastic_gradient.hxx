/**     
*/
#ifndef INFERNO_LEARNING_LEARNERS_STOCHASTIC_GRADIENT_HXX
#define INFERNO_LEARNING_LEARNERS_STOCHASTIC_GRADIENT_HXX


#include "inferno/learning/learners/learners.hxx"
#include "inferno/utilities/index_vector.hxx"
#include "inferno/utilities/line_search/line_search.hxx"
#include "inferno/inference/base_discrete_inference_factory.hxx"


#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <fstream>

namespace inferno{
namespace learning{
namespace learners{




    template<class DATASET>
    class StochasticGradient{
    public:
        typedef DATASET Dataset;
        typedef typename Dataset::Model         Model;
        typedef typename Dataset::GroundTruth   GroundTruth;
        typedef typename Dataset::LossFunction  LossFunction;
        typedef typename Model:: template VariableMap<DiscreteLabel> ConfMap;
        typedef std::vector<ConfMap> ConfMapVector;
        typedef inference::BaseDiscreteInferenceFactory<Model> InferenceFactoryBase;

        struct Options{
            Options(
                const uint64_t nPertubations = 100,
                const uint64_t maxIterations = 10000,
                const double   sigma = 1.0,
                const int      verbose =2,
                const int      seed = 0,
                const double   n = 1.0
            )
            :   nPertubations_(nPertubations),
                maxIterations_(maxIterations),
                sigma_(sigma),
                verbose_(verbose),
                seed_(seed),
                n_(n)
            {
            }

            uint64_t nPertubations_;
            uint64_t maxIterations_;
            double   sigma_;
            int      verbose_;
            int seed_;
            double n_;
        };

        StochasticGradient(Dataset & dset, const Options & options = Options())
        :   dataset_(dset),
            options_(options){
        }

        Dataset & dataset(){
            return dataset_;
        }

        void learn(InferenceFactoryBase * inferenceFactory, WeightVector & weightVector){

            try{

                double TOL = 1e-8;
                int lastImprove = 0;

                std::ofstream output("/home/argo/HCI/bachelorarbeit/output.txt");

    

                // get dataset
                auto & dset = dataset();
                //auto & weights = weightVector;
                WeightVector prevStep(weightVector.size(),0);
                
                bestLoss_  = dataset_.averageLoss(inferenceFactory);
                currentLoss_ = bestLoss_;
                std::cout << "initial Loss: " << bestLoss_ << std::endl;
                WeightVector bestWeight = weightVector;

                // multiple weight-vectors stacked as matrix
                WeightMatrix            noiseMatrix(weightVector,options_.nPertubations_);
                WeightMatrix            weightMatrix(weightVector,options_.nPertubations_);   
                std::vector<LossType>   losses(options_.nPertubations_);

                std::vector<LossType>   relLosses(options_.nPertubations_);


                // random gen
                boost::mt19937 rng(options_.seed_); // I don't seed it on purpouse (it's not relevant)
                boost::normal_distribution<> nd(0.0, options_.sigma_);
                boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > normalDist(rng, nd);



                // indices
                utilities::IndexVector< > indices(dset.size());


                const auto & weightConstraints =  dset.weightConstraints();

                for(size_t i=0; i<options_.maxIterations_; ++i){
                   
                    std::cout << "Iteration " << i << "\n";

                    indices.randomShuffle();
                    // FIXME indices.randomShuffle(rng);

                    for(const auto trainingInstanceIndex : indices){

                        std::cout << "Image " << trainingInstanceIndex << "\n";

                        // unlock model
                        dset.unlock(trainingInstanceIndex);

                        //std::cout<<"fetch "<<trainingInstanceIndex<<"\n";
                        // get model, gt, and loss-function
                        auto & model = *dset.model(trainingInstanceIndex);
                        const auto & gt = *dset.groundTruth(trainingInstanceIndex);
                        const auto  lossFunction = dset.lossFunction(trainingInstanceIndex);

                        // pertubate (and remember noise matrix)
                        weightMatrix.pertubate(weightVector,noiseMatrix,weightConstraints, normalDist); 

                        // to remember arg mins
                        //std::cout<<"conf assign \n";
                        ConfMapVector confMapVector(options_.nPertubations_);
                        for(auto & cmap : confMapVector){
                            cmap.assign(model);
                        }

                        // get loss unperturbed weights
                        model.updateWeights(weightVector);

                        // get argmin 
                        auto inference = inferenceFactory->create(model);
                        inference->infer();
                        inference->conf(confMapVector[0]); 

                        const auto loss_unperturbed = lossFunction->eval(model, gt, confMapVector[0]);


                        // argmin for perturbed model
                        WeightVector gradient(weightVector.size(),0);
                        bool directImprove = false;
                        auto directImproveIndex = 0;
                        LossType directImproveValue = infVal();
                        auto cc=0;
                        for(const auto & perturbedWeightVector : weightMatrix){
                            

                            // set perturbed weights
                            model.updateWeights(perturbedWeightVector);

                            // get argmin 
                            auto inference = inferenceFactory->create(model);
                            inference->infer();
                            inference->conf(confMapVector[cc]); 

                            const auto l = lossFunction->eval(model, gt, confMapVector[cc]);
                            const auto relLoss = l - loss_unperturbed;
                            relLosses[cc] = relLoss * loss_unperturbed;
                            //std::cout << "relLoss: " << relLoss << "\n";

                            if (l < loss_unperturbed) {
                                directImprove = true;
                                

                                if (l < directImproveValue) {
                                    directImproveIndex = cc;
                                    directImproveValue = l;
                                }


                                //std::cout << "Local direct Improve found at Iteration " << i << " with relLoss " << relLoss 
                                //          << "  Potential: " << -relLoss / loss_unperturbed << "%\n";

                            }

                            losses[cc] = l;
                            ++cc;

                        }
                        
                        // Just use the perturbed weightVector which yielded the biggest decrease in loss of current training sample
                        /*if (directImprove) {
                            for (size_t k=0; k<relLosses.size(); ++k) {

                                if (k != directImproveIndex) {
                                    relLosses[k] = 0;
                                }
                                else {
                                    relLosses[k] *= options_.nPertubations_;
                                }

                            } 
                        }*/
                        

                        //noiseMatrix.weightedSum(losses, gradient); // original absolute-weighted method
                        noiseMatrix.weightedSum(relLosses, gradient);

                        gradient *= dset.regularizer().c()/double(options_.nPertubations_);


                        // reset the weights to the current weights
                        model.updateWeights(weightVector);


                        takeGradientStep(inferenceFactory, weightVector, gradient, prevStep, 
                                         bestWeight, i);


                        // Save current weights in external file
                        for (const auto& w : weightVector) {
                            output << std::fixed << std::setw(5) << w << "\t";
                        }
                        output << "\n";


                        dset.updateWeights(weightVector);
                        

                        // lock
                        dset.lock(trainingInstanceIndex);
                    }

                    if (bestLoss_ < TOL) {    // Konvergenzkriterium
                        std::cout << "bestLoss smaller than tolerance!\n";
                        break;
                    }


                }                
                weightVector = bestWeight;
                dset.updateWeights(weightVector);


                output.close();

            }
            catch( const std::exception &e) { 
                throw RuntimeError(std::string("an Error happened in StochasticGradient:\n") + e.what());
            }
            catch(...){
                throw; //RuntimeError("an Error happened in StochasticGradient:\n");
            }
        }
    private:



        bool takeGradientStep(
            InferenceFactoryBase * inferenceFactory,
            WeightVector & currentWeights,
            WeightVector & gradient,
            WeightVector & prevStep,
            WeightVector & bestWeight,
            const uint64_t iteration
        ){
            double it(iteration+1);
            //WeightVector newWeights = currentWeights;

      
            const auto  effectiveStepSize = options_.n_/(it);

            auto takeStep = [&,this] (const double stepSize, bool undo = true, bool eval = true){

                WeightVector buffer = currentWeights;

                WeightVector g=gradient;
                g*=stepSize;
                currentWeights -= g;

                double momentumStr = 0.25;
                WeightVector m=prevStep;
                m*=momentumStr;
                currentWeights += m;

                prevStep = g;
                prevStep *= -1.;
                prevStep += m;

                //currentWeights *= options_.alpha_;

                
                // fix bounded weights
                const auto & wConstraints = dataset_.weightConstraints();
                for(const auto kv : wConstraints.weightBounds()){
                    const auto wi = kv.first;
                    const auto lowerBound = kv.second.first;
                    const auto upperBound = kv.second.second;
                    if(currentWeights[wi] < lowerBound){
                       currentWeights[wi] = lowerBound; 
                    }
                    if(currentWeights[wi] > upperBound){
                       currentWeights[wi] = upperBound; 
                    }
                }
                
                dataset_.updateWeights(currentWeights);
                LossType loss = 0;
                if(eval)
                    loss = dataset_.averageLoss(inferenceFactory);


                    //std::cout << "Current Weights:  ";
                    /*for (const auto& cw : currentWeights) {
                        std::cout << cw << "   ";
                    }*/
                    //std::cout << currentWeights[0] << "   " << currentWeights[0] << "...\n";
                    //std::cout << "\nAverageLoss:\t" << loss << "\n";


                if(undo)
                    currentWeights = buffer;
                return loss;
            };

            //std::cout<<"take step"<<takeStep(effectiveStepSize,false,true)<<"\n";


            std::vector<double> fracs({10.0, 5.0, 1.0, 0.5, 0.1});
            std::vector<double> lossVal(fracs.size());

            
            bool improvment = false;

            int bestIndex = 0;
            double bestVal = infVal();

            

            for(size_t i=0; i<fracs.size(); ++i){
                const double ss = effectiveStepSize*fracs[i];
                const double ll = takeStep(ss);
                lossVal[i] = ll;

                if(ll<bestLoss_){
                    std::cout << "improved via frac  " << fracs[i] << "     loss " << ll << "  at Iteration: " << iteration << "\n";
                    bestLoss_ = ll;
                    currentLoss_ = bestLoss_;
                    takeStep(ss, false, false);
                    bestWeight = currentWeights;
                    improvment = true;
                    bestVal = ll;
                    bestIndex = i;

                    // Save current bestWeight
                    std::ofstream backupBestWeightsFile("/home/argo/HCI/bachelorarbeit/sgBackupWeights.txt");
                    for (const auto& w : bestWeight) {
                        backupBestWeightsFile << w << "\n";
                    }
                    backupBestWeightsFile.close();


                    return true;        // wahrscheinlich auch suboptimal ohne die anderen zu probieren.. 
                }
                if(ll<bestVal){
                    bestVal = ll;
                    bestIndex = i;
                }
            }
            if(!improvment){
                currentLoss_  = lossVal[bestIndex];
                //std::cout<<"best via frac  "<<fracs[bestIndex]<<" loss "<<lossVal[bestIndex]<<"\n";
                
             

                /*for (const auto& lv : lossVal) {
                    std::cout << lv << "   ";
                }
                std::cout << "\n";*/


                const double ss = effectiveStepSize*fracs[bestIndex];
                takeStep(ss, false, false);
            }
            return false;
        }

        Dataset & dataset_;
        Options options_;

        LossType currentLoss_;
        LossType bestLoss_;  
    };

} // end namespace inferno::learning::learners
} // end namespace inferno::learning
} // end namespace inferno


#endif /* INFERNO_LEARNING_LEARNERS_STOCHASTIC_GRADIENT_HXX */
