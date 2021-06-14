using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace XSSDefender.ConsoleClient
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            // TrainModel();

            UseTrainedModel();
        }

        private static void UseTrainedModel()
        {
            // 1. MLContext
            var context = new MLContext();

            // 2. Load model
            var model = context.Model.Load("xss-model.zip", out _);

            // 3. Engine
            var engine = context.Model.CreatePredictionEngine<XssInput, XssPrediction>(model);

            // 4. Predict

            XssInput input1 = new XssInput
            {
                Sentence = "<input type=hidden name=foo value=&gt;&lt;script&#32;src=http://attacker/ bad.js&gt;&lt;/script&gt;>"
            };

            XssInput input2 = new XssInput
            {
                Sentence = "<input type=text name=firstname value=&quot;&gt;&lt;Hello World&gt;>"
            };

            XssInput input3 = new XssInput
            {
                Sentence = "Hello World <script alert('hello')>"
            };

            var prediction = engine.Predict(input3);

            Console.WriteLine($"{prediction.Sentence} {prediction.Prediction} {prediction.Probability:P2}");

        }

        private static void TrainModel()
        {
            // 1. MLContext
            var context = new MLContext();

            // 2. Load data

            TextLoader.Options options = new TextLoader.Options
            {
                HasHeader = true,
                AllowQuoting = true,
                Separators = new char[] { ',' }
            };

            // var model = context.Data.LoadFromTextFile<XssInput>("XSS_dataset.csv", separatorChar: ',', hasHeader: true, allowQuoting: true);

            var dataView = context.Data.LoadFromTextFile<XssInput>("XSS_dataset.csv", options);

            var preview = dataView.Preview();

            var split = context.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var trainingData = split.TrainSet;
            var testData = split.TestSet;

            // 3. Transformacja

            var pipeline = context.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(XssInput.Sentence));

            var trainer = context.BinaryClassification.Trainers.SdcaLogisticRegression();

            var trainingPipeline = pipeline.Append(trainer);

            // 4. Training

            Console.WriteLine("Training model...");
            var model = trainingPipeline.Fit(trainingData);
            Console.WriteLine("Trained.");

            // 5. Evalue
            Console.WriteLine("Evaluating...");
            var predictions = model.Transform(testData);

            var metrics = context.BinaryClassification.Evaluate(predictions);
            Console.WriteLine($"Evaluated. {metrics.Accuracy:P2}");

            // 6. Save model
            context.Model.Save(model, trainingData.Schema, "xss-model.zip");

        }
    }

    public class XssInput
    {
        [LoadColumn(1)]
        public string Sentence { get; set; }

        [LoadColumn(2)]
        public bool Label { get; set; }
    }

    public class XssPrediction : XssInput
    {
        [ColumnName("PredictatedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
