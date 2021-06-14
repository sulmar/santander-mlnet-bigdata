using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace TextSentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            // dotnet add package Microsoft.ML

            // TrainModel();

            UseTrainedModel();

        }

        private static void UseTrainedModel()
        {
            // 1. Utworzenie kontekstu (Create Context)
            MLContext context = new MLContext();

            // 2. Ładowanie modelu
            var model = context.Model.Load("text-sentiment-model.zip", out _);

            // 3. Utworzenie instancji silnika predykcji
            var engine = context.Model.CreatePredictionEngine<SentimentInput, SentimentPrediction>(model);

            // 4. Predykcja

            bool loop = true;

            Console.CancelKeyPress += (sender, args) => loop = false;

            while (loop)
            {
                Console.Write("> ");
                var line = Console.ReadLine();
                SentimentInput sentimentInput = new SentimentInput { Text = line };
                var prediction = engine.Predict(sentimentInput);
                Console.WriteLine($"{prediction.Prediction} {prediction.Probability}");
            }

        }

        private static void TrainModel()
        {
            // 1. Utworzenie kontekstu (Create Context)
            MLContext context = new MLContext(seed: 1);

            // 2. Ładowanie danych (Load data)
            IDataView dataView = context.Data.LoadFromTextFile<SentimentInput>("wikipedia-detox-250-line-data.tsv", hasHeader: true);

            var preview = dataView.Preview();

            var trainTestSplit = context.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var trainingData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;


            // 3. Przetwarzanie danych (Transform data)
            var pipeline = context.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(SentimentInput.Text));

            // 4. Wybór algorytmu (Choose Algorithm)

            // Sdca
            var trainer = context.BinaryClassification.Trainers.SdcaLogisticRegression();

            var processPipeline = pipeline.Append(trainer);

            // 5. Uczenie modelu (Train model)
            Console.WriteLine("Training model...");

            var trainedModel = processPipeline.Fit(trainingData);

            Console.WriteLine("Trained.");

            // 6. Ocena modelu (Evaluate model)
            Console.WriteLine("Evaluating model...");

            var predictions = trainedModel.Transform(testData);

            var metrics = context.BinaryClassification.Evaluate(predictions);

            Console.WriteLine($"{metrics.Accuracy:P2}");

            // 7. Wdrożenie i konsumpcja modelu (Deploy & Consume)

            context.Model.Save(trainedModel, trainingData.Schema, "text-sentiment-model.zip");
        }
    }

    public class SentimentInput
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }
    }

    public class SentimentPrediction : SentimentInput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
