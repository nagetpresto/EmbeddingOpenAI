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
        #region Input Parameter
        var keyword = "Paracetamol drop";
        double threshold = 80;

        int offsetCatalog = 0;
        const int catalogBatchSize = 1000;
        #endregion

        var proc = "";
        List<CatalogData> allCatalogData = new List<CatalogData>();
        var results = new List<Result>();

        Console.WriteLine("start");
        Console.WriteLine("");
        Console.WriteLine($"keyword: {keyword}");
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
            #endregion

            #region request embedding
            proc = "request embedding";
            var textsBatch = new List<string> { keyword.ToString() };
            var embeddings = await GetEmbeddingsAsync(apiKey, endpoint, textsBatch);
            #endregion

            #region calculate similarity
            proc = "calculate similarity";
            for (int i = 0; i < embeddings.Count; i++)
            {
                var topSimilarities = CalculateTopSimilarities(embeddings[i], allCatalogData, threshold);

                int index = 0;
                foreach (var eachSimilatiry in topSimilarities)
                {
                    var resultDetail = await GetDetailsByMasterID(connection, eachSimilatiry.MasterID);

                    var resultHeader = await GetHeaderByMasterID(connection, eachSimilatiry.MasterID);

                    eachSimilatiry.ResultDetail = resultDetail;
                    eachSimilatiry.MasterCode = resultHeader[0].MasterCode;
                    eachSimilatiry.MasterName = resultHeader[0].MasterName;
                    eachSimilatiry.Type = resultHeader[0].Type;
                    eachSimilatiry.ID = index +1;

                    index++;
                }
                    
                results.Add(new Result
                {
                    ID = i + 1,
                    Description = keyword.ToString(),
                    TopSimilarities = topSimilarities
                });
            }

            #endregion

            #region console result
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine("result:");
            string jsonResults = JsonConvert.SerializeObject(results, Formatting.Indented);
            Console.WriteLine(jsonResults);

            Console.WriteLine("");
            Console.WriteLine("done");
            #endregion

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

    private static List<SimilarityResult> CalculateTopSimilarities(string embedding, List<CatalogData> masterCatalogData, double threshold)
    {
        var currentEmbedding = Vector<double>.Build.DenseOfArray(JsonConvert.DeserializeObject<double[]>(embedding));
        var topSimilarities = new List<SimilarityResult>();
        int index = 0;

        foreach (var catalogItem in masterCatalogData)
        {
            var masterEmbedding = Vector<double>.Build.DenseOfEnumerable(catalogItem.Embedding);

            double dotProduct = currentEmbedding.DotProduct(masterEmbedding);
            double magnitudeA = currentEmbedding.L2Norm();
            double magnitudeB = masterEmbedding.L2Norm();

            double similarity = (magnitudeA > 0 && magnitudeB > 0) ? (dotProduct / (magnitudeA * magnitudeB)) * 100 : 0;

            if (similarity > threshold)
            {
                topSimilarities.Add(new SimilarityResult
                {
                    
                    Similarity = similarity,
                    MasterID = catalogItem.MasterID
                });

                topSimilarities = topSimilarities
                    .OrderByDescending(result => result.Similarity)
                    .Take(5)
                    .ToList();

                index++;
            }
        }

        return topSimilarities;
    }

    private static async Task<List<ResultDetail>> GetDetailsByMasterID(SqlConnection connection, int masterID)
    {
        var details = new List<ResultDetail>();

        try
        {
            var commandText = "SELECT TOP 5 * FROM OpenAI.dbo.MasterDrugs WHERE FolariumID = " + masterID.ToString() + " ORDER BY Name asc, Qty desc ";
            
            using (var command = new SqlCommand(commandText, connection))
            {
                
                using (var reader = await command.ExecuteReaderAsync())
                {
                    int i = 0;
                    while (await reader.ReadAsync())
                    {
                        details.Add(new ResultDetail
                        {
                            ID = i + 1,
                            DetailID = reader.GetInt32(0),
                            Name = reader.IsDBNull(2) ? null : reader.GetString(2),
                            Brand = reader.IsDBNull(3) ? null : reader.GetString(3),
                            Manufaktur = reader.IsDBNull(4) ? null : reader.GetString(4),
                            Qty = reader.IsDBNull(5) ? null : reader.GetDecimal(5),
                            Unit = reader.IsDBNull(6) ? null : reader.GetString(6),
                            Unit_Detail = reader.IsDBNull(7) ? null : reader.GetString(7),
                            Price = reader.IsDBNull(8) ? null : reader.GetDecimal(8),
                        });

                        i++;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"Error retrieving details for MasterID {masterID}: {ex.Message}");
            throw;
        }

        return details.ToList();
    }

    private static async Task<List<ResultHeader>> GetHeaderByMasterID(SqlConnection connection, int masterID)
    {
        var header = new List<ResultHeader>();

        try
        {
            var commandText = "SELECT * FROM OpenAI.dbo.MasterFolarium WHERE ID = " + masterID.ToString() ;
            
            using (var command = new SqlCommand(commandText, connection))
            {

                using (var reader = await command.ExecuteReaderAsync())
                {
                    int i = 0;
                    while (await reader.ReadAsync())
                    {
                        header.Add(new ResultHeader
                        {
                            ID = i + 1,
                            MasterCode = reader.IsDBNull(1) ? null : reader.GetString(1),
                            MasterName = reader.IsDBNull(2) ? null : reader.GetString(2),
                            Type = reader.IsDBNull(3) ? null : reader.GetString(3),
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
    public int MasterID { get; set; }
    public List<double> Embedding { get; set; }
}

public class Result
{
    public int ID { get; set; }
    public string Description { get; set; }
    public List<SimilarityResult> TopSimilarities { get; set; } = new List<SimilarityResult>();
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
    public int MasterID { get; set; }
    public string MasterCode { get; set; }
    public string MasterName { get; set; }
    public string Type { get; set; }
    public double Similarity { get; set; }
    public List<ResultDetail> ResultDetail { get; set; } = new List<ResultDetail>();
}

public class ResultDetail
{
    public int ID { get; set; }
    public int DetailID { get; set; }
    public string Name { get; set; }
    public string? Brand { get; set; }
    public string? Manufaktur { get; set; }
    public decimal? Qty { get; set; }
    public string? Unit { get; set; }
    public string? Unit_Detail { get; set; }
    public decimal? Price { get; set; }
}

public class ResultHeader
{
    public int ID { get; set; }
    public string MasterCode { get; set; }
    public string MasterName { get; set; }
    public string Type { get; set; }
}


