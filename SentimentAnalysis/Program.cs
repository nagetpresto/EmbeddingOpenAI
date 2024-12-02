using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using Microsoft.Data.SqlClient;
using System.Data;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.Security.Claims;
using System.Drawing;

class Program
{
    static async Task Main(string[] args)
    {
        try
        {
            var config = LoadConfiguration();
            var connectionString = config.GetConnectionString("DefaultConnection");
            var openAiApiKey = config["OpenAI_Embedding:ApiKey"];
            var openAiEndpointEmbedding = config["OpenAI_Embedding:EndpointEmbedding"];
            var openAiEndpointGPT4 = config["OpenAI_Embedding:EndpointGPT4"];

            using (var connection = new SqlConnection(connectionString))
            {
                await connection.OpenAsync();
                await ProcessBatchesAsync(connection, openAiApiKey, openAiEndpointEmbedding, openAiEndpointGPT4);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }

    private static IConfiguration LoadConfiguration()
    {
        return new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: false)
            .Build();
    }

    private static async Task ProcessBatchesAsync(SqlConnection connection, string apiKey, string endpointEmbedding, string endpointGPT4)
    {
        #region Input Parameter
        var reply = "rpa lu jelek";
        double threshold = 80;

        int offsetCatalog = 0;
        const int catalogBatchSize = 1000;
        #endregion

        var proc = "";
        List<CatalogData> allCatalogData = new List<CatalogData>();
        var results = new List<Result>();

        Console.WriteLine("start");
        Console.WriteLine("");
        Console.WriteLine($"keyword: {reply}");
        Console.WriteLine($"threshold: {threshold}");

        try
        {
            #region get catalog
            proc = "get catalog";
            while (true)
            {
                var catalogData = await GetMasterCatalogEmbedding(connection, offsetCatalog, catalogBatchSize);
                if (catalogData.Count == 0) break;

                allCatalogData.AddRange(catalogData);

                offsetCatalog += catalogBatchSize;
            }

            Console.WriteLine($"catalog count: {allCatalogData.Count}");
            Console.WriteLine("");
            #endregion

            #region request sentiment
            proc = "request sentiment";
            var prompt = $"Analyze the sentiment of the following review (consider all languages, slangs, and capitalization): {reply} .Answer in json with key 'sentiment', 'product_name', 'reason'";
            var (messageContent, totalTokens, promptTokens, completionTokens) = await GetResponseGPT4(apiKey, endpointGPT4, prompt);
            Console.WriteLine("sentiment result:");
            Console.WriteLine(messageContent);
            Console.WriteLine("");

            Console.WriteLine("total token");
            Console.WriteLine(totalTokens);
            Console.WriteLine("prompt token");
            Console.WriteLine(promptTokens);
            Console.WriteLine("completion token");
            Console.WriteLine(completionTokens);

            var messageData = JsonConvert.DeserializeObject<dynamic>(messageContent);

            string sentiment = messageData.sentiment;
            string productName = messageData.product_name;
            string reason = messageData.reason;
            #endregion

            #region request embedding product
            proc = "request embedding";
            var textsBatch = new List<string> { productName.ToString() };
            var embeddings = await GetEmbeddingsAsync(apiKey, endpointEmbedding, textsBatch);
            #endregion

            if (sentiment == "negative")
            {
                #region calculate similarity
                proc = "calculate similarity";
                for (int i = 0; i < embeddings.Count; i++)
                {
                    int bestMasterID = 0;
                    var bestSimilarity = 0.0;
                    var bestName = "";

                    // cek ke catalog per batch
                    var similarityResult = CalculateSimilarity(embeddings[i], allCatalogData, bestSimilarity, bestMasterID, threshold, bestName);
                    bestSimilarity = similarityResult.Similarity;
                    bestMasterID = similarityResult.ID;
                    bestName = similarityResult.ProductName;

                    // add to result
                    results.Add(new Result
                    {
                        Reply = reply,
                        Sentiment = sentiment,
                        Reason = reason,
                        inputToken = promptTokens,
                        ouputToken = completionTokens,
                        ProductReply = productName,
                        ProductID = bestMasterID,
                        ProductName = bestName,
                        Similarity = bestSimilarity,
                    });

                }
                #endregion

                #region console result similarity
                Console.WriteLine("");
                Console.WriteLine("result similarity:");
                string jsonResults = JsonConvert.SerializeObject(results, Formatting.Indented);
                Console.WriteLine(jsonResults);

                Console.WriteLine("");
                #endregion

                #region request create reply
                proc = "request create reply";
                int productID = results[0].ProductID == 0 ? 1 : results[0].ProductID;
                var productKeywords = await GetProductKeywords(connection, productID);
                var keywordsText = string.Join(",", productKeywords.ConvertAll(item => item.Keywords));

                prompt = $"user reply: '{reply}' . construct a reply for negative reply sent by user using these keywords {keywordsText}. tell them that my company and my company product is useful and how you can help them. make it short for social media reply, and at the end give this contact details sales@elistec.com. use the same language as the user reply but with formal style.";

                Console.WriteLine("promt");
                Console.WriteLine(prompt);
                Console.WriteLine("");

                (messageContent, totalTokens, promptTokens, completionTokens) = await GetResponseGPT4(apiKey, endpointGPT4, prompt);
                Console.WriteLine("result reply:");
                Console.WriteLine(messageContent);
                Console.WriteLine("");

                Console.WriteLine("total token");
                Console.WriteLine(totalTokens);
                Console.WriteLine("prompt token");
                Console.WriteLine(promptTokens);
                Console.WriteLine("completion token");
                Console.WriteLine(completionTokens);
                #endregion

            }

            else
            {
                Console.WriteLine("setiment is not negative");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"Error processing at {proc}; offsetCatalog {offsetCatalog}: {ex.Message}");
        }
    }

    private static async Task<List<CatalogData>> GetMasterCatalogEmbedding(SqlConnection connection, int offset, int batchSize)
    {
        var catalogItems = new List<CatalogData>();

        try
        {
            using (var command = new SqlCommand("GetCatalogEmbedding", connection))
            {
                command.CommandType = CommandType.StoredProcedure;
                command.Parameters.AddWithValue("@Offset", offset);
                command.Parameters.AddWithValue("@BatchSize", batchSize);

                using (var reader = await command.ExecuteReaderAsync())
                {
                    while (await reader.ReadAsync())
                    {
                        var embeddingJson = reader.GetString(2);
                        var embedding = JsonConvert.DeserializeObject<List<double>>(embeddingJson);

                        catalogItems.Add(new CatalogData
                        {
                            ID = reader.GetInt32(0),
                            ProductName = reader.GetString(1),
                            Embedding = embedding
                        });

                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"Error retrieving catalog data: {ex.Message}");
            throw;
        }

        return catalogItems;
    }

    private static async Task<(string MessageContent, int TotalTokens, int PromptTokens, int CompletionTokens)> GetResponseGPT4(string apiKey, string endpoint, string texts)
    {
        try
        {
            using (var httpClient = new HttpClient())
            {
                httpClient.DefaultRequestHeaders.Add("api-key", apiKey);

                var requestBody = new
                {
                    model = "gpt-4",
                    messages = new[]
                    {
                        new
                        {
                            role = "user",
                            content = texts
                        }
                    },
                    temperature = 0,
                    max_tokens = 100
                };

                var jsonContent = JsonConvert.SerializeObject(requestBody);
                var response = await httpClient.PostAsync(
                    endpoint,
                    new StringContent(jsonContent, Encoding.UTF8, "application/json")
                );

                response.EnsureSuccessStatusCode();
                var result = await response.Content.ReadAsStringAsync();

                var responseObject = JsonConvert.DeserializeObject<dynamic>(result);

                string messageContent = responseObject.choices[0].message.content;
                int totalTokens = responseObject.usage.total_tokens;
                int promptTokens = responseObject.usage.prompt_tokens;
                int completionTokens = responseObject.usage.completion_tokens;

                return (messageContent, totalTokens, promptTokens, completionTokens);
            }

        }
        catch (HttpRequestException ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"HTTP Request error: {ex.Message}");
            throw;
        }
        catch (JsonException ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"JSON parsing error: {ex.Message}");
            throw;
        }
    }

    private static async Task<List<string>> GetEmbeddingsAsync(string apiKey, string endpoint, List<string> texts)
    {
        try
        {
            using (var httpClient = new HttpClient())
            {
                httpClient.DefaultRequestHeaders.Add("api-key", apiKey);
                var jsonContent = JsonConvert.SerializeObject(new { input = texts });
                var response = await httpClient.PostAsync(endpoint, new StringContent(jsonContent, Encoding.UTF8, "application/json"));
                response.EnsureSuccessStatusCode();

                var result = await response.Content.ReadAsStringAsync();
                var embeddingsResponse = JsonConvert.DeserializeObject<EmbeddingsResponse>(result);
                return embeddingsResponse.data.ConvertAll(embedding => JsonConvert.SerializeObject(embedding.embedding));
            }
        }
        catch (HttpRequestException ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"HTTP Request error: {ex.Message}");
            throw;
        }
        catch (JsonException ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"JSON parsing error: {ex.Message}");
            throw;
        }
    }

    private static SimilarityResult CalculateSimilarity(string embedding, List<CatalogData> masterCatalogData, double bestSimilarity, int bestMasterID, double threshold, string bestName)
    {
        var currentEmbedding = Vector<double>.Build.DenseOfArray(JsonConvert.DeserializeObject<double[]>(embedding));
        double tmpSimilarity = bestSimilarity;
        int tmpMasterID = bestMasterID;
        var tmpName = bestName;

        foreach (var catalogItem in masterCatalogData)
        {
            var masterEmbedding = Vector<double>.Build.DenseOfEnumerable(catalogItem.Embedding);

            double dotProduct = currentEmbedding.DotProduct(masterEmbedding);
            double magnitudeA = currentEmbedding.L2Norm();
            double magnitudeB = masterEmbedding.L2Norm();

            double similarity = (magnitudeA > 0 && magnitudeB > 0) ? (dotProduct / (magnitudeA * magnitudeB)) * 100 : 0;

            if ((similarity > threshold) && (similarity > tmpSimilarity))
            {
                tmpSimilarity = similarity;
                tmpMasterID = catalogItem.ID;
                tmpName = catalogItem.ProductName;
            }
        }

        return new SimilarityResult
        {
            Similarity = tmpSimilarity,
            ID = tmpMasterID,
            ProductName = tmpName
        };
    }

    private static async Task<List<ProductKeywords>> GetProductKeywords(SqlConnection connection, int masterID)
    {
        var header = new List<ProductKeywords>();


        try
        {
            var commandText = "select Keywords from OpenAI.stm.MasterKnowledgeProduct where ProductID = " + masterID.ToString();

            using (var command = new SqlCommand(commandText, connection))
            {

                using (var reader = await command.ExecuteReaderAsync())
                {
                    int i = 0;
                    while (await reader.ReadAsync())
                    {
                        header.Add(new ProductKeywords
                        {
                            Keywords = reader.IsDBNull(0) ? null : reader.GetString(0)
                        });

                        i++;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"Error retrieving header for MasterID {masterID}: {ex.Message}");
            throw;
        }

        return header.ToList();
    }

}

public class CatalogData
{
    public int ID { get; set; }
    public string ProductName { get; set; }
    public List<double> Embedding { get; set; }
}

public class Result
{
    public string Reply { get; set; }
    public string Sentiment { get; set; }
    public string Reason { get; set; }
    public int inputToken { get; set; }
    public int ouputToken { get; set; }
    public string ProductReply { get; set; }
    public int ProductID { get; set; }
    public string ProductName { get; set; }
    public double Similarity { get; set; }
}

public class EmbeddingsResponse
{
    public List<EmbeddingData> data { get; set; }
}

public class EmbeddingData
{
    public List<double> embedding { get; set; }
}

public class SimilarityResult
{
    public int ID { get; set; }
    public string ProductName { get; set; }
    public double Similarity { get; set; }
}

public class ProductKeywords
{
    public string Keywords { get; set; }
}
