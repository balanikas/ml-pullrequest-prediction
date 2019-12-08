using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace MLTest
{
    class Program
    {
        static readonly HttpClient client = new HttpClient();
        private static readonly Random rng = new Random();

        static async Task Main(string[] args)
        {
            var prData = await DownloadPullRequestDataAsync("bitbucket-rest-api-endpoint-base");
            SaveJson(prData);

            var rawJson = LoadRawJson();
            var lines = new List<string>();
            foreach (var x in rawJson)
            {
               lines.Add( CreateCsvLine(x));
            }

            Shuffle(lines);

            var trainingData = lines.Take(lines.Count / 2);
            var testData = lines.Skip(lines.Count / 2);

            SaveCsv("trainingdata", trainingData);
            SaveCsv("testdata", testData);
        }

        static async Task<JArray> DownloadPullRequestDataAsync(string endpoint)
        {
            var authToken = Convert.ToBase64String(Encoding.UTF8.GetBytes("user:password"));
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", authToken);

            JArray raw = new JArray();

            await foreach (var projectPage in Get($"{endpoint}projects"))
            {
                foreach (var p in projectPage["values"].AsJEnumerable())
                {
                    var projectKey = p["key"].ToString();

                    await foreach (var repoPage in Get($"{endpoint}projects/{projectKey}/repos"))
                    {
                        foreach (var r in repoPage["values"].AsJEnumerable())
                        {
                            var repoName = r["name"].ToString();

                            await foreach (var prPage in Get($"{endpoint}projects/{projectKey}/repos/{repoName}/pull-requests", "&state=Merged"))
                            {
                                foreach (var pr in prPage["values"].AsJEnumerable())
                                {
                                    raw.Add(pr);
                                }
                            }
                        }
                    }
                }
            }

            return raw;
        }

        public static void Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        private static async IAsyncEnumerable<JObject> Get(string resource, string query = null)
        {
            int nextPageStart = 0;
            bool isLastPage;
            do
            {
                var response = await client.GetAsync($"{resource}?limit=100&start={nextPageStart}{query}");
                response.EnsureSuccessStatusCode();
                var content = JObject.Parse(await response.Content.ReadAsStringAsync());
                isLastPage = bool.Parse(content.SelectToken("isLastPage").Value<string>());
                if (!isLastPage) 
                    nextPageStart = int.Parse(content.SelectToken("nextPageStart").Value<string>());
                yield return content;
            }
            while (!isLastPage);
        }

        public static string CreateCsvLine(JToken token)
        {
            var closed = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
            closed = closed.AddMilliseconds(long.Parse(token["closedDate"].ToString())).ToLocalTime();

            var started = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
            started = started.AddMilliseconds(long.Parse(token["createdDate"].ToString())).ToLocalTime();

            var lifeTime = Math.Round((closed - started).TotalHours, 0);

            var line = string.Join(",",
                token["links"].SelectToken("self").Single().SelectToken("href").ToString(),
                token["fromRef"].SelectToken("repository").SelectToken("project").SelectToken("key").ToString(),
                token["fromRef"].SelectToken("repository").SelectToken("slug").ToString(),
                lifeTime,
                token["author"].SelectToken("user").SelectToken("name").ToString(),
                token["reviewers"].Count() + token["participants"].Count(),
                token["properties"].SelectToken("commentCount")?.ToString() ?? "0",
                token["properties"].SelectToken("resolvedTaskCount")?.ToString() ?? "0",
                token["description"]?.ToString().Length.ToString() ?? "0",
                token["fromRef"].SelectToken("displayId").ToString().Contains("feature/") ? "1" : "0");

            return line;
        }

        public static void SaveCsv(string fileName, IEnumerable<string> lines)
        {
            var headers = string.Join(",",
                "link",
                "project",
                "repo",
                "lifetime",
                "author",
                "reviewersCount",
                "commentCount",
                "resolvedTasks",
                "descriptionLength",
                "branch");

            File.Delete(Path.Combine(Environment.CurrentDirectory, $"{fileName}.csv"));

            File.AppendAllText(Path.Combine(Environment.CurrentDirectory, $"{fileName}.csv"), headers + "\n" + string.Join("\n", lines));
        }

        public static void SaveJson(JArray arr)
        {
            File.Delete(Path.Combine(Environment.CurrentDirectory, "raw.json"));
            File.AppendAllText(Path.Combine(Environment.CurrentDirectory, "raw.json"), arr.ToString());
        }

        public static dynamic LoadRawJson() => 
            JsonConvert.DeserializeObject<dynamic>(File.ReadAllText(Path.Combine(Environment.CurrentDirectory, "out.json")));
    }
}
