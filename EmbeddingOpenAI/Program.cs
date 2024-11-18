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

class Program
{
    static async Task Main(string[] args)
    {
        try
        {
            var config = LoadConfiguration();
            var connectionString = config.GetConnectionString("DefaultConnection");
            var openAiApiKey = config["OpenAI_Embedding:ApiKey"];
            var openAiEndpoint = config["OpenAI_Embedding:Endpoint"];

            using (var connection = new SqlConnection(connectionString))
            {
                await connection.OpenAsync();
                await ProcessBatchesAsync(connection, openAiApiKey, openAiEndpoint);
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

    private static async Task ProcessBatchesAsync(SqlConnection connection, string apiKey, string endpoint)
    {
        int claimID = 0;
        double threshold = 0; 
        int batch = 1;
        int offset = 0;
        const int batchSize = 100;

        int offsetCatalog = 0;
        const int catalogBatchSize = 1000;

        List<CatalogData> allCatalogData = new List<CatalogData>();
        var results = new List<Result>();

        Console.WriteLine("start");
        Console.WriteLine("");
        Console.WriteLine($"claimID: {claimID}");

        try
        {
            while (true)
            {
                // get catalog
                while (true)
                {
                    var catalogData = await GetMasterCatalogEmbedding(connection, offsetCatalog, catalogBatchSize);
                    if (catalogData.Count == 0) break;

                    allCatalogData.AddRange(catalogData);

                    offsetCatalog += catalogBatchSize;
                }                

                // get invoice
                var invoiceData = await GetInvoiceDataAsync(connection, offset, batchSize, claimID);
                if (invoiceData.Count == 0) break;

                Console.WriteLine($"catalog count: {allCatalogData.Count}");
                Console.WriteLine($"invoice count: {invoiceData.Count}");
                Console.WriteLine($"batch: {batch}");

                // collect invoice description
                var idsBatch = invoiceData.ConvertAll(item => item.ID);
                var textsBatch = invoiceData.ConvertAll(item => item.Description);

                // request embedding
                var embeddings = await GetEmbeddingsAsync(apiKey, endpoint, textsBatch);

                // calculating similarity
                for (int i = 0; i < embeddings.Count; i++)
                {
                    int bestMasterID = 0;
                    var bestSimilarity = 0.0;
                    var invID = invoiceData[i].ID;
                    var invDescription = invoiceData[i].Description;

                    Console.WriteLine($"cek => {invDescription}");

                    // cek ke catalog per batch
                    var similarityResult = CalculateSimilarity(embeddings[i], allCatalogData, bestSimilarity, bestMasterID, threshold);
                    bestSimilarity = similarityResult.Similarity;
                    bestMasterID = similarityResult.MasterID;

                    // add to result
                    results.Add(new Result
                    {
                        ID = invoiceData[i].ID,
                        Description = invoiceData[i].Description,
                        SimilarityPercentage = bestSimilarity,
                        MasterID = bestMasterID
                    });

                }

                batch++;
                offset += batchSize;
            }

            // get name in mater catalog
            await GetMasterNameAsync(connection, results);

            // console result
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine("result:");
            Console.WriteLine("ID | Description | Similarity | MasterCode | MasterID | MasterName");
            foreach (var result in results)
            {
                Console.WriteLine($"{result.ID} | {result.Description} | {result.SimilarityPercentage} | {result.MasterCode} | {result.MasterID} | {result.MasterName}");
            }

            Console.WriteLine("");
            Console.WriteLine("done");

        }
        catch (Exception ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"Error processing batch {batch}, offset {offset}: {ex.Message}");
        }

    }

    private static async Task<List<Invoice>> GetInvoiceDataAsync(SqlConnection connection, int offset, int batchSize, int claimID)
    {
        var invoiceData = new List<Invoice>();

        try
        {
            using (var command = new SqlCommand("GetInvoices", connection))
            {
                command.CommandType = CommandType.StoredProcedure;
                command.Parameters.AddWithValue("@Offset", offset);
                command.Parameters.AddWithValue("@BatchSize", batchSize);
                command.Parameters.AddWithValue("@ClaimID", claimID);

                using (var reader = await command.ExecuteReaderAsync())
                {
                    while (await reader.ReadAsync())
                    {
                        invoiceData.Add(new Invoice
                        {
                            ID = reader.GetInt32(0),
                            Description = reader.GetString(1)
                        });
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"Error retrieving invoice data: {ex.Message}");
            throw;
        }

        return invoiceData;
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
                            MasterID = reader.GetInt32(1),
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

    private static SimilarityResult CalculateSimilarity(string embedding, List<CatalogData> masterCatalogData, double bestSimilarity, int bestMasterID, double threshold)
    {
        var currentEmbedding = Vector<double>.Build.DenseOfArray(JsonConvert.DeserializeObject<double[]>(embedding));
        double tmpSimilarity = bestSimilarity;
        int tmpMasterID = bestMasterID;

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
                tmpMasterID = catalogItem.MasterID;
            }
        }

        return new SimilarityResult
        {
            Similarity = tmpSimilarity,
            MasterID = tmpMasterID
        };
    }

    //private static SimilarityResult CalculateSimilarity(string embedding, List<CatalogData> masterCatalogData, double bestSimilarity, int bestMasterID, double threshold)
    //{
    //    var currentEmbedding = JsonConvert.DeserializeObject<List<double>>(embedding);
    //    double tmpSimilarity = bestSimilarity;
    //    int tmpMasterID = bestMasterID;

    //    foreach (var catalogItem in masterCatalogData)
    //    {
    //        var masterEmbedding = catalogItem.Embedding;

    //        double dotProduct = currentEmbedding.Zip(masterEmbedding, (a, b) => a * b).Sum();
    //        double magnitudeA = Math.Sqrt(currentEmbedding.Sum(x => x * x));
    //        double magnitudeB = Math.Sqrt(masterEmbedding.Sum(x => x * x));

    //        double similarity = (magnitudeA > 0 && magnitudeB > 0) ? (dotProduct / (magnitudeA * magnitudeB)) * 100 : 0;

    //        if ((similarity > threshold) && (similarity > tmpSimilarity))
    //        {
    //            tmpSimilarity = similarity;
    //            tmpMasterID = catalogItem.MasterID;
    //        }
    //    }

    //    return new SimilarityResult
    //    {
    //        Similarity = tmpSimilarity,
    //        MasterID = tmpMasterID
    //    };
    //}

    private static async Task GetMasterNameAsync(SqlConnection connection, List<Result> results)
    {
        try
        {
            var masterIds = results.Select(r => r.MasterID).Distinct().ToList();

            var commandText = "SELECT ID, Code, Name FROM OpenAI.dbo.MasterFolarium WHERE ID IN (" + string.Join(",", masterIds) + ")";

            using (var command = new SqlCommand(commandText, connection))
            {
                using (var reader = await command.ExecuteReaderAsync())
                {
                    var masterNameMap = new Dictionary<int, (string Name, string Code)>();
                    while (await reader.ReadAsync())
                    {
                        var id = reader.GetInt32(0);
                        var code = reader.GetString(1);
                        var name = reader.GetString(2);
                        masterNameMap[id] = (Name: name, Code: code);
                    }

                    foreach (var result in results)
                    {
                        if (masterNameMap.TryGetValue(result.MasterID, out var masterInfo))
                        {
                            result.MasterName = masterInfo.Name;
                            result.MasterCode = masterInfo.Code;
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"Error retrieving master names: {ex.Message}");
        }
    }

}

public class Invoice
{
    public int ID { get; set; }
    public string Description { get; set; }
}

public class CatalogData
{
    public int ID { get; set; }
    public int MasterID { get; set; }
    public List<double> Embedding { get; set; }
}

public class Result
{
    public int ID { get; set; }
    public string Description { get; set; }
    public int MasterID { get; set; }
    public string MasterCode { get; set; }
    public string MasterName { get; set; }

    public double SimilarityPercentage { get; set; }
}

public class SimilarityResult
{
    public double Similarity { get; set; }
    public int MasterID { get; set; }
}

public class EmbeddingsResponse
{
    public List<EmbeddingData> data { get; set; }
}

public class EmbeddingData
{
    public List<double> embedding { get; set; }
}