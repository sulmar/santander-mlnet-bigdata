using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.IO;

namespace MovieRecommendation
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            // Context
            var context = new MLContext();

            // Load data
            var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            IDataView trainingDataView = context.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = context.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            // Transform

            var pipeline = context.Transforms.Conversion.MapValueToKey(
                inputColumnName: nameof(MovieRating.userId),
                outputColumnName: "userIdEncoded")
                .Append(context.Transforms.Conversion.MapValueToKey(
                    inputColumnName: nameof(MovieRating.movieId),
                    outputColumnName: "movieIdEncoded"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            // dotnet add package Microsoft.ML.Recommender

            var trainer = pipeline.Append(context.Recommendation().Trainers.MatrixFactorization(options));


            // Train model

            var trainedModel = trainer.Fit(trainingDataView);

            // Evaluate

            var prediction = trainedModel.Transform(testDataView);

            var metrics = context.Regression.Evaluate(prediction);

            Console.WriteLine($"{metrics.RSquared}");

            // Prediction

            var engine = context.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(trainedModel);


            var input = new MovieRating { userId = 1, movieId = 10 };

            var movieRatingPrediction = engine.Predict(input);


            if (movieRatingPrediction.Score > 3.5f)
            {
                Console.WriteLine($"Movie {input.movieId} is recommended for {input.userId} with score {movieRatingPrediction.Score}");
            }
            else
            {
                Console.WriteLine($"Movie {input.movieId} is NOT recommended for {input.userId} with score {movieRatingPrediction.Score}");
            }


         
        }
    }

    // userId,movieId,rating,timestamp

    internal class MovieRating
    {
        [LoadColumn(0)]
        public float userId { get; set; }
        [LoadColumn(1)] 
        public float movieId { get; set; }
        [LoadColumn(2)] 
        public float Label { get; set; }
    }

    public class MovieRatingPrediction
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }
}
