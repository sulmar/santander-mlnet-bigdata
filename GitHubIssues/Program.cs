using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO.Compression;

namespace GitHubIssues
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            const string filename = "./extract/corefx-issues-train.tsv";

            // Unzip
            if (!System.IO.File.Exists(filename))
                ZipFile.ExtractToDirectory("corefx-issues-train.zip", "./extract");


            // 1. Context
            var context = new MLContext();

            // 2. Load data
            // IDataView dataView = context.Data.LoadFromTextFile<GitHubIssue>(filename, hasHeader: true);

            TextLoader loader = context.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column("ID", DataKind.String, 0),
                new TextLoader.Column("Area", DataKind.String, 1),
                new TextLoader.Column("Title", DataKind.String, 2),
                new TextLoader.Column("Description", DataKind.String, 3),

            },
            hasHeader: true
            );

            

            IDataView dataView = loader.Load(filename);


            var preview = dataView.Preview();

            // 3. Transform data

            // Zamiana wartości na klucz 
            var pipeline = context.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(GitHubIssue.Area), outputColumnName: "Label");

            var featurizedTitle = context.Transforms.Text.FeaturizeText(inputColumnName: nameof(GitHubIssue.Title), outputColumnName: "TitleFeaturized");
            var featurizedDescription = context.Transforms.Text.FeaturizeText(inputColumnName: nameof(GitHubIssue.Description), outputColumnName: "DescriptionFeaturized");

            var features = context.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: new string[] { "TitleFeaturized", "DescriptionFeaturized" });
            pipeline.Append(featurizedTitle);
            pipeline.Append(featurizedDescription);
            pipeline.Append(features);

            // 4. Algorithm

            var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy();

            pipeline.Append(trainer);

            // Zamiana klucza na wartość
            var mapKeyToValue = context.Transforms.Conversion.MapKeyToValue("PredictedLabel");

            pipeline.Append(mapKeyToValue);

            // 5. Train model
            var trainedModel = pipeline.Fit(dataView);

            // 6. Evaluate model
            //var metrics = context.MulticlassClassification.Evaluate(trainedModel.Transform(dataView));

            //Console.WriteLine($"{metrics.MicroAccuracy}");
            //Console.WriteLine($"{metrics.MacroAccuracy}");
            //Console.WriteLine($"{metrics.LogLoss}");
            //Console.WriteLine($"{metrics.LogLossReduction}");

            // 7. Predict

            var engine = context.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(trainedModel);

            GitHubIssue issue = new GitHubIssue
            {
                Title = "WebSocket communication is slow in my machine",
                Description = "WebSocket communication use SignalR on my development machine."
            };

            var prediction = engine.Predict(issue);

            Console.WriteLine($"{prediction.Area}");

        }
    }

    public class GitHubIssue
    {
        public string ID { get; set; }
        public string Area { get; set; }
        public string Title { get; set; }
        public string Description { get; set; }
    }

    //public class GitHubIssue
    //{
    //    [LoadColumn(0)]
    //    public string ID { get; set; }
    //    [LoadColumn(1)]
    //    public string Area { get; set; }
    //    [LoadColumn(2)]
    //    public string Title { get; set; }
    //    [LoadColumn(3)]
    //    public string Description { get; set; }
    //}

    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area { get; set; }
        public float Score { get; set; }
    }
}
