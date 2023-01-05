import matplotlib.pyplot as plt
import neptune.new as neptune

run = neptune.init_run(project="WideLearning/Titanic",
                       api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
                       with_id="TIT-9",
                       mode="read-only",
                       )

# print(dir(run))
for name, series in run.get_structure().items():
    if isinstance(series, dict):
        continue
    print(name)
    data = series.fetch_values()["value"].to_numpy()
    plt.plot(data, label=name)
plt.legend()
plt.show()