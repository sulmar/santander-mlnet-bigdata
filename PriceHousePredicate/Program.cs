using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLFlow.NET.Lib;
using MLFlow.NET.Lib.Contract;
using MLFlow.NET.Lib.Model;
using MLFlow.NET.Lib.Model.Responses.Run;
using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace PriceHousePredicate
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            // OnnxSaveTest();

            await TrainModel();
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

        private static async Task TrainModel()
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

            var services = _bootApp();

            var flowService = services.GetService<IMLFlowService>();

            var response = await flowService.GetOrCreateExperiment("Experiment-MLNET-MLFlow");

            
           //  var response = await flowService.CreateExperiment("Experiment-MLNET4");
            var experimentId = response.ExperimentId;

            var createRunRequest = new CreateRunRequest()
            {
                ExperimentId = experimentId,
                UserId = Environment.UserName,
                SourceType = SourceType.NOTEBOOK,
                SourceName = "String descriptor for the run’s source",
                EntryPointName = "Name of the project entry point associated with the current run, if any.",
                StartTime = ((DateTimeOffset)DateTime.UtcNow).ToUnixTimeMilliseconds() //unix timestamp
            };

            var runResult = await flowService.CreateRun(createRunRequest);

            var runUuid = runResult.Run.Info.RunUuid;


            //IProgress<RunDetail<RegressionMetrics>> progress = 
            //    new Progress<RunDetail<RegressionMetrics>>(d => Console.WriteLine($"{d.TrainerName} R^2 {d.ValidationMetrics?.RSquared} {d.RuntimeInSeconds} s"));


            var progress = new FlowMLProgress(flowService, runUuid);

            Console.WriteLine("Starting experiment. Press Ctrl+C to cancel.");

            var result = experiment.Execute(dataView, labelColumnName: "Label", progressHandler: progress);

            Console.WriteLine("Finished experiment.");

            Console.WriteLine($"The best trainer {result.BestRun.TrainerName} R^2 {result.BestRun.ValidationMetrics.RSquared} {result.BestRun.RuntimeInSeconds} s");

            var trainedModel = result.BestRun.Model;

            context.Model.Save(trainedModel, null, $"{result.BestRun.TrainerName}-model.zip");

            return;

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


        static IServiceProvider _bootApp()
        {
            var builder = new ConfigurationBuilder();
            // tell the builder to look for the appsettings.json file
            builder.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);

            var configuration = builder.Build();

            var serviceCollection = new ServiceCollection();

            serviceCollection.AddMFlowNet();

            serviceCollection.Configure<MLFlowConfiguration>(
                configuration.GetSection(nameof(MLFlowConfiguration)
                ));

            var serviceProvider = serviceCollection.BuildServiceProvider();
            return serviceProvider;
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
