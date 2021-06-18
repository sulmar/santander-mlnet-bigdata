using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using MLFlow.NET.Lib;
using MLFlow.NET.Lib.Contract;
using MLFlow.NET.Lib.Model;
using MLFlow.NET.Lib.Model.Responses.Run;

namespace MLFlowConsoleClient
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            var services = _bootApp();

            var flowService = services.GetService<IMLFlowService>();


            // var response = await flowService.CreateExperiment("Experiment2");

            //var experimentId = response.ExperimentId;

            //Console.WriteLine($"{response.ExperimentId}");


            var experimentId = 2;

            var createRunRequest = new CreateRunRequest()
            {
                ExperimentId = experimentId,
                UserId = "John Smith",
                SourceType = SourceType.NOTEBOOK,
                SourceName = "String descriptor for the run’s source",
                EntryPointName = "Name of the project entry point associated with the current run, if any.",
                StartTime = ((DateTimeOffset)DateTime.UtcNow).ToUnixTimeMilliseconds() //unix timestamp
            };

            var runResult = await flowService.CreateRun(createRunRequest);

            var runUuid = runResult.Run.Info.RunUuid;

            var logResultMetric = await flowService
              .LogMetric(runUuid, "Foo", 2345);

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
}
