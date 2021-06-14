using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using XSSDefender.ConsoleClient;

namespace XSSDefender.WebApp.Middlewares
{
    public class AntiXssMiddleware
    {
        private readonly RequestDelegate next;

        // dotnet add package Microsoft.Extensions.ML
        private readonly PredictionEnginePool<XssInput, XssPrediction> predictionEnginePool;
        private readonly ILogger<AntiXssMiddleware> logger;

        public AntiXssMiddleware(RequestDelegate next, PredictionEnginePool<XssInput, XssPrediction> predictionEnginePool, ILogger<AntiXssMiddleware> logger)
        {
            this.next = next;
            this.predictionEnginePool = predictionEnginePool;
            this.logger = logger;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            context.Request.EnableBuffering();

            using (var reader = new StreamReader(context.Request.Body, Encoding.UTF8, false, 1024, true))
            {
                var body = await reader.ReadToEndAsync();

                XssInput input = new XssInput { Sentence = body };

                var prediction = predictionEnginePool.Predict(input);

                if (prediction.Prediction)
                {

                    logger.LogError($"{context.Request.Method} {prediction.Sentence} {prediction.Prediction} {prediction.Probability:P2}");

                    context.Response.StatusCode = StatusCodes.Status400BadRequest;
                    await context.Response.WriteAsync("XSS atack detected!");

                    // zerwanie połączenia
                    // context.Abort();

                    return;
                }
                else
                {
                    logger.LogInformation($"{context.Request.Method} {prediction.Sentence} {prediction.Prediction} {prediction.Probability:P2}");
                }


            }

            context.Request.Body.Seek(0, SeekOrigin.Begin);

            await next.Invoke(context);
        }
    }
}
