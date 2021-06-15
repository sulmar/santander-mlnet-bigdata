using Microsoft.ML;
using System;
using System.Collections.Generic;

namespace TextTransformationConsoleClient
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            IEnumerable<SentimentModel> messages = new SentimentModel[]
            {
                new SentimentModel { Text = "Hello World" },
                new SentimentModel { Text = "Lorem ipsum" },
                new SentimentModel { Text = "Hello ML.NET" },
                new SentimentModel { Text = "Hello NET Core" },
            };

            // 1. Context
            var context = new MLContext();

            // 2. Load data
            var dataView = context.Data.LoadFromEnumerable<SentimentModel>(messages);

            var preview = dataView.Preview();



        }
    }

    public class SentimentModel
    {
        public string Text { get; set; }
    }
}
