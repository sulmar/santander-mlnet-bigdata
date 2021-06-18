using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLFlow.NET.Lib.Contract;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PriceHousePredicate
{
    public class FlowMLProgress : IProgress<RunDetail<RegressionMetrics>>
    {
        private readonly IMLFlowService flowService;
        private readonly string runUuid;

        public FlowMLProgress(IMLFlowService flowService, string runUuid)
        {
            this.flowService = flowService;
            this.runUuid = runUuid;
        }

        public async Task ReportAsync(RunDetail<RegressionMetrics> value)
        {
          
                // await flowService.LogParameter(runUuid, "TrainerName", value.TrainerName);

                 await flowService.LogMetric(runUuid, $"{value.TrainerName}-Foo", 2345);


            
            //mfloat runtimeInSeconds = (float) value.RuntimeInSeconds;

           await flowService.LogMetric(runUuid, $"{value.TrainerName}-RuntimeInSeconds", 1.001f);

        }

        public void Report(RunDetail<RegressionMetrics> value)
        {
            Console.WriteLine($"{value.TrainerName} R^2 {value.ValidationMetrics?.RSquared} {value.RuntimeInSeconds} s");


            ReportAsync(value).Wait();


            

            //if (value.ValidationMetrics!=null)
            //    flowService.LogMetric(runUuid, "R2", (float) value.ValidationMetrics?.RSquared);

            //flowService.LogMetric(runUuid, "RuntimeInSeconds", (float)value.RuntimeInSeconds);


        }
    }
}
