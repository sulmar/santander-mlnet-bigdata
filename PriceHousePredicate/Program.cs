using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;


namespace PriceHousePredicate
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            // OnnxSaveTest();

            TrainModel();
        }

        private static void OnnxSaveTest()
        {
            var context = new MLContext();

            const string filename = "./extract/Houses.csv";

            var model = context.Model.Load("FastTreeRegression-model.zip", out _);

            IDataView dataView = context.Data.LoadFromTextFile<HousingData>(filename, hasHeader: true, separatorChar: ',');

            using (var onnx = File.Open("model.onnx", FileMode.OpenOrCreate))
            {
                context.Model.ConvertToOnnx(model, dataView, onnx);
            }
        }


        // dotnet add package Microsoft.ML.AutoML

        private static void TrainModel()
        {
            const string filename = "./extract/Houses.csv";

            // Unzip
            if (!System.IO.File.Exists(filename))
                ZipFile.ExtractToDirectory("Houses.csv.zip", "./extract");


            // 1. Context
            var context = new MLContext();

            // 2. Load data
            IDataView dataView = context.Data.LoadFromTextFile<HousingData>(filename, hasHeader: true, separatorChar: ',');

            var cts = new CancellationTokenSource();

            Console.CancelKeyPress += (sender, args) => cts.Cancel();

            var experimentSettings = new RegressionExperimentSettings
            {
                // MaxExperimentTimeInSeconds = (uint) TimeSpan.FromMinutes(15).TotalSeconds,

                MaxExperimentTimeInSeconds = 10,
                OptimizingMetric = RegressionMetric.RSquared,
                CacheDirectory = new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, "Models")),
                CancellationToken = cts.Token,
            };

            // istnieje możliwość usunięcia trenerów z eksperymentu
            experimentSettings.Trainers.Remove(RegressionTrainer.LbfgsPoissonRegression);

            var experiment = context.Auto().CreateRegressionExperiment(experimentSettings);

            IProgress<RunDetail<RegressionMetrics>> progress = 
                new Progress<RunDetail<RegressionMetrics>>(d => Console.WriteLine($"{d.TrainerName} R^2 {d.ValidationMetrics?.RSquared} {d.RuntimeInSeconds} s"));

            Console.WriteLine("Starting experiment. Press Ctrl+C to cancel.");

            var result = experiment.Execute(dataView, labelColumnName: "Label", progressHandler: progress);

            Console.WriteLine("Finished experiment.");

            Console.WriteLine($"The best trainer {result.BestRun.TrainerName} R^2 {result.BestRun.ValidationMetrics.RSquared} {result.BestRun.RuntimeInSeconds} s");

            var trainedModel = result.BestRun.Model;

            context.Model.Save(trainedModel, null, $"{result.BestRun.TrainerName}-model.zip");

            var details = result.RunDetails
                .OrderByDescending(m => m.ValidationMetrics.RSquared)
                .Take(3);

            foreach (var detail in details)
            {
                if (detail.ValidationMetrics!=null)
                    Console.WriteLine($"Other trainer {detail.TrainerName} R^2 {detail.ValidationMetrics.RSquared} {detail.RuntimeInSeconds} s");

                //var trainedModel = detail.Model;

                //context.Model.Save(trainedModel, null, $"{detail.TrainerName}-model.zip");


            }

        }
    }

    internal class HousingData
    {
        //[ColumnName("lp"), LoadColumn(0)]
        //public int Lp { get; set; }
        [ColumnName("address"), LoadColumn(1)]
        public string Address { get; set; }
        [ColumnName("city"), LoadColumn(2)]
        public string City { get; set; }
        [ColumnName("floor"), LoadColumn(3)]
        public float Floor { get; set; }
        [ColumnName("id"), LoadColumn(4)]
        public float Id { get; set; }
        [ColumnName("latitude"), LoadColumn(5)]
        public float Latitude { get; set; }
        [ColumnName("longitude"), LoadColumn(6)]
        public float Longitude { get; set; }
        [ColumnName("Label"), LoadColumn(7)] // <--- Label
        public float Price { get; set; }
        [ColumnName("rooms"), LoadColumn(8)]
        public float Rooms { get; set; }
        [ColumnName("sq"), LoadColumn(9)]
        public float Sq { get; set; }
        [ColumnName("year"), LoadColumn(10)]
        public float Year { get; set; }

    }
}
