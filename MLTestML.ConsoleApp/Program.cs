using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLTestML.Model;

namespace MLTestML.ConsoleApp
{
    class Program
    {
        //Dataset to use for predictions 
        private const string TRAINING_DATA_FILEPATH = @"trainingdata.csv";
        private const string TEST_DATA_FILEPATH = @"testdata.csv";
        private static string ModelPath = @"out.zip";

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            BuildTrainEvaluateAndSaveModel(mlContext);

            TestSinglePrediction(mlContext);
        }

        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            IDataView baseTrainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(TRAINING_DATA_FILEPATH, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<ModelInput>(TEST_DATA_FILEPATH, hasHeader: true, separatorChar: ',');

            IDataView trainingDataView = mlContext.Data.FilterRowsByColumn(baseTrainingDataView, nameof(ModelInput.Lifetime), lowerBound: 0, upperBound: 1000);
            trainingDataView = mlContext.Data.FilterRowsByColumn(trainingDataView, nameof(ModelInput.ResolvedTasks), lowerBound: 0, upperBound: 30);
            trainingDataView = mlContext.Data.FilterRowsByColumn(trainingDataView, nameof(ModelInput.CommentCount), lowerBound: 0, upperBound: 100);

            var dataProcessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(ModelInput.Lifetime))
                 .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ProjectEncoded", inputColumnName: nameof(ModelInput.Project)))
                 .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RepoEncoded", inputColumnName: nameof(ModelInput.Repo)))
                 .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AuthorEncoded", inputColumnName: nameof(ModelInput.Author)))
                 .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(ModelInput.CommentCount)))
                 .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(ModelInput.DescriptionLength)))
                 .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(ModelInput.ResolvedTasks)))
                 .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(ModelInput.ReviewersCount)))
                 .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(ModelInput.Branch)))
                 .Append(mlContext.Transforms.Concatenate("Features",
                    "ProjectEncoded",
                    "RepoEncoded",
                    "AuthorEncoded",
                    nameof(ModelInput.CommentCount) , 
                    nameof(ModelInput.DescriptionLength), 
                    nameof(ModelInput.ResolvedTasks),
                    nameof(ModelInput.ReviewersCount),
                    nameof(ModelInput.Branch)
                    ));

            ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 5);
            ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", trainingDataView, dataProcessPipeline, 5);

            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");

            IDataView predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            
            ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return trainedModel;
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            var sample = new ModelInput()
            {
                CommentCount = 100,
                ResolvedTasks = 2,
                ReviewersCount = 1,
                DescriptionLength = 100,
                Project = "??",
                Repo = "???",
                Lifetime = 0 
            };

            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out _);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
            var resultprediction = predEngine.Predict(sample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted lifetime: {resultprediction.Score:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
