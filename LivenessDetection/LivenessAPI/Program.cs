using AIModels.Detection;
using AIModels.Landmark;
using AIModels.Recognition;
using AIModels.Spoof;
using LivenessAPI.Application;
using LivenessAPI.Application.Abstractions;
using LivenessAPI.Infrastructure.ChallengeEngine;
using LivenessAPI.Infrastructure.FaceDetection;
using LivenessAPI.Infrastructure.FaceLandmark;
using LivenessAPI.Infrastructure.FaceRecognition;
using LivenessAPI.Infrastructure.PassiveLiveness;
using LivenessAPI.Infrastructure.Session;

var builder = WebApplication.CreateBuilder(args);

const string CorsPolicy = "LivenessCors";

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.Configure<LivenessOptions>(builder.Configuration.GetSection(LivenessOptions.SectionName));
builder.Services.AddMemoryCache();

// Dev-permissive CORS so a browser-based camera client on a different origin can call the API.
// Tighten this (named origins, no AllowAnyHeader) before deploying anywhere public.
builder.Services.AddCors(options =>
{
    options.AddPolicy(CorsPolicy, policy => policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
});

// ONNX-backed detectors: each wraps a single InferenceSession and is safe to share as a singleton
// (ONNX Runtime's Run() is thread-safe for concurrent calls on the same session).
builder.Services.AddSingleton(sp => new ScrfdDetector(ResolveModelPath(sp, o => o.ScrfdModelPath)));
builder.Services.AddSingleton(sp => new Landmark106Detector(ResolveModelPath(sp, o => o.LandmarkModelPath)));
builder.Services.AddSingleton(sp => new ArcFaceRecognizer(ResolveModelPath(sp, o => o.RecognitionModelPath)));
builder.Services.AddSingleton(sp => new AntiSpoofDetector(ResolveModelPath(sp, o => o.AntiSpoofModelPath)));

builder.Services.AddSingleton<IFaceDetectionService, FaceDetectionService>();
builder.Services.AddSingleton<IFaceLandmarkService, FaceLandmarkService>();
builder.Services.AddSingleton<IAntiSpoofService, AntiSpoofService>();
builder.Services.AddSingleton<IFaceRecognitionService, FaceRecognitionServiceImpl>();
builder.Services.AddSingleton<IChallengeEngine, ChallengeEngineImpl>();
builder.Services.AddSingleton<ILivenessSessionStore, MemoryLivenessSessionStore>();
builder.Services.AddScoped<ILivenessSessionService, LivenessSessionService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseDefaultFiles();
app.UseStaticFiles();

app.UseCors(CorsPolicy);

app.UseAuthorization();

app.MapControllers();

app.Run();

static string ResolveModelPath(IServiceProvider sp, Func<LivenessOptions, string> selector)
{
    var options = sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<LivenessOptions>>().Value;
    var env = sp.GetRequiredService<IWebHostEnvironment>();
    var relative = selector(options);
    var resolved = Path.GetFullPath(Path.Combine(env.ContentRootPath, relative));

    if (!File.Exists(resolved))
    {
        throw new FileNotFoundException($"Liveness model file not found at '{resolved}'. Check the Liveness config section in appsettings.json.", resolved);
    }

    return resolved;
}
