using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TextSentimentAnalysis;

namespace TextSentimentAnalysisWebApi.Controllers
{
    [Route("api")]
    public class PredictionController : ControllerBase
    {
        // POST localhost:5000/api/predict

        // dotnet add package Microsoft.Extensions.ML

        private readonly PredictionEnginePool<SentimentInput, SentimentPrediction> predictionEnginePool;

        public PredictionController(PredictionEnginePool<SentimentInput, SentimentPrediction> predictionEnginePool)
        {
            this.predictionEnginePool = predictionEnginePool;
        }

        [Route("Predict")]
        public ActionResult<SentimentPrediction> Predict([FromBody] string text)
        {
            var sentimentInput = new SentimentInput { Text = text };

            var prediction = predictionEnginePool.Predict(sentimentInput);

            return prediction;
        }
    }
}
