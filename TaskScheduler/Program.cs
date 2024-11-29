using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using Microsoft.Data.SqlClient;
using System.Data;

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
        int batch = 1;
        int offset = 0;
        const int batchSize = 100;

        Console.WriteLine("start");
        Console.WriteLine("");

        while (true)
        {
            try
            {
                // get catalog
                var catalogData = await GetCatalogDataAsync(connection, offset, batchSize);
                if (catalogData.Count == 0) break;

                Console.WriteLine($"batch: {batch}, offset: {offset}");

                // collect catalog name
                var textsBatch = catalogData.ConvertAll(item => item.ProductName);
                
                // request embedding
                var embeddings = await GetEmbeddingsAsync(apiKey, endpoint, textsBatch);

                // save embedding
                await SaveEmbeddingsBatchAsync(connection, catalogData, embeddings);
                
                batch++;
                offset += batchSize;
            }
            catch (Exception ex)
            {
                Console.WriteLine("");
                Console.WriteLine($"Error processing batch {batch}, offset {offset}: {ex.Message}");
                break;
            }
        }
    }

    private static async Task<List<CatalogItem>> GetCatalogDataAsync(SqlConnection connection, int offset, int batchSize)
    {
        var catalogItems = new List<CatalogItem>();

        try
        {
            using (var command = new SqlCommand("GetCatalogItems", connection))
            {
                command.CommandType = CommandType.StoredProcedure;
                command.Parameters.AddWithValue("@Offset", offset);
                command.Parameters.AddWithValue("@BatchSize", batchSize);

                using (var reader = await command.ExecuteReaderAsync())
                {
                    while (await reader.ReadAsync())
                    {
                        catalogItems.Add(new CatalogItem
                        {
                            ID = reader.GetInt32(0),
                            ProductName = reader.GetString(1)
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

    private static async Task SaveEmbeddingsBatchAsync(SqlConnection connection, List<CatalogItem> catalogData, List<string> embeddings)
    {
        var dataTable = new DataTable();
        dataTable.Columns.Add("MasterID", typeof(int));
        dataTable.Columns.Add("Embedding", typeof(string));

        for (int i = 0; i < catalogData.Count; i++)
        {
            dataTable.Rows.Add(catalogData[i].ID, embeddings[i]);
        }

        try
        {
            using (var command = new SqlCommand("SaveEmbeddingsBatch", connection))
            {
                command.CommandType = CommandType.StoredProcedure;
                var parameter = command.Parameters.AddWithValue("@EmbeddingsTable", dataTable);
                parameter.SqlDbType = SqlDbType.Structured;
                await command.ExecuteNonQueryAsync();
            }

            Console.WriteLine("saved");
            Console.WriteLine("");
        }
        catch (Exception ex)
        {
            Console.WriteLine("");
            Console.WriteLine($"Error saving embeddings: {ex.Message}");
            throw;
        }
    }
}

public class CatalogItem
{
    public int ID { get; set; }
    public string ProductName { get; set; }
}

public class EmbeddingsResponse
{
    public List<EmbeddingData> data { get; set; }
}

public class EmbeddingData
{
    public List<double> embedding { get; set; }
}
