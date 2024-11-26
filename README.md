# Run your personal MLFlow server on the Apolo platform with apolo-flow

This is a [`apolo-flow`](https://github.com/neuro-inc/neuro-flow) action launching an instance of [MLFlow server](https://www.mlflow.org/docs/latest/tracking.html).
You can use it to track your ML experiments and model trainings, track model in model registry and perform model deployment to production using our integrations [in-job-deployments](https://github.com/neuro-inc/mlops-job-deploy-app) or [MLFlow2Seldon integration](https://github.com/neuro-inc/mlops-k8s-mlflow2seldon), or build own integrations.

The MLFlow action exposes several arguments, one of which is mandatory: `artifacts_destination`.

## Usage example could be found in the [.neuro/live.yaml](.neuro/live.yaml) file.

## Arguments
### `mode`

Mode of operation of MLFlow server. Allowed values are "server" or "ui".

- `server` mode. The MLFLow server should run while MLFlow Client is connected to it. Refer to it as to **Scenario 5** from [official MLFlow docs](https://www.mlflow.org/docs/latest/tracking.html#scenario-5-mlflow-tracking-server-enabled-with-proxied-artifact-storage-access). This mode is also required to run in-job ML models deployments.

- `ui` mode. In this case, MLFlow server only serves the artifacts and metadata previously stored by clients on a backend store. See [official MLFlow docs](https://www.mlflow.org/docs/latest/cli.html#mlflow-ui) describing this use-case. The bennefit of it is that you should not run MLFlow server all the time while the training happens. This helps to save costs and HW resources in a constrained environment.

#### Example

Running server in UI mode:

```yaml
args:
    mode: "ui"
```


### `artifacts_destination`

A local path within the MLFlow server, whee to store artifacts such as model dumps.
You can find more information [here](https://mlflow.org/docs/latest/tracking.html#artifact-stores)

#### Example

You can use platform storage as a backend.
To do this, use a local path for artifact store:
1. Set this input's value to the mount path of the needed volume.
2. Add its read-write reference to the `inputs.volumes` list.

```yaml
args:
    artifacts_destination: ${{ volumes.mlflow_artifacts.mount }}
    volumes: "${{ to_json( [volumes.mlflow_artifacts.ref_rw] ) }}"
```


### `backend_store_uri`

URI of the storage which should be used to dump experiments metainfo, their metrics, registered models, etc.
You can find more information [here](https://mlflow.org/docs/latest/tracking.html#backend-stores).

#### Examples

* The argument is not set.
In this case the `--backend_store_uri` MLFlow flag will be ommited and the default value will be used (see the _regular file_ case below).

* Postgres server as a job within the same project:
```yaml
args:
    backend_store_uri: postgresql://postgres:password@${{ inspect_job('postgres').internal_hostname_named }}:5432
```

* SQLite persistent on a platform disk or storage.
This also implies adding the respective disk's or volume reference to the `volumes` argument.
```yaml
args:
    artifacts_destination: ${{ volumes.mlflow_artifacts.mount }}
    backend_store_uri: sqlite:///${{ volumes.mlflow_artifacts.mount }}/mlflow.db
    volumes: "${{ to_json( [ volumes.mlflow_artifacts.ref_rw ] ) }}"
```

* Regular file. 
In this case, the *MLFlow registered models* functionality will not work.
```yaml
args:
    backend_store_uri: /path/to/store 
```

### `volumes`

Reference to a list of volumes which should be mounted to the MLFlow server job. Empty by default.

#### Example

```yaml
args:
    volumes: "${{ to_json(
        [
          volumes.mlflow_artifacts.ref_rw,
          volumes.mlflow_storage.ref_rw
        ]
      ) }}"
```

### `envs`

List of environment variables added to the job. Empty by default.

#### Example

```yaml
args:
    envs: "${{ to_json(
        {
          'ENV1': 'env_1_value',
          'ENV2': 'env_2_value'
        }
      ) }}"
```

### `http_auth`

Boolean value specifying whether to use HTTP authentication for Jupyter or not. `"False"` by default.

#### Example

Enable HTTP authentication by setting this argument to True.
```yaml
args:
    http_auth: "True"
```

_**Note**: your training job should be able to communicate with MLFlow guarded by the Apolo platform authentication solution. In order to do so, you should put a token of a user (or a service account), which has access the corresponding MLFlow server, into the `MLFLOW_TRACKING_TOKEN` environment variable within the training job._

### `life_span`

A value specifying how long the MLFlow server job should be running. `"10d"` by default.

#### Example

```yaml
args:
    life_span: 1d2h3m
```

### `port`

HTTP port to use for the MLFlow server. `"5000"` by default.

#### Example

```yaml
args:
    http_port: "4444"
```

### `job_name`

Predictable subdomain name which replaces the job's ID in the full job URI. `""` by default (i.e., the job ID will be used).

#### Example

```yaml
args:
    job_name: "mlflow-server"
```


### `preset`

Resource preset to use when running the Jupyter job. `""` by default (i.e., the first preset specified in the `apolo config show` list will be used).

#### Example

```yaml
args:
    preset: cpu-small
```

### `extra_params`

Additional parameters transferred to the `mlflow server` command. `""` by default.
Check the full list of accepted parameters via `mlflow server --help`.

#### Example

```yaml
args:
    extra_params: "--workers 2"
```

# Known issues
### `sqlite3.OperationalError: database is locked`
This might happen under the following circumstances:
1. the mlflow server parameter `--backend_store_uri` is not set (by default, SQLite is used) or set to use SQLite or a regular file
2. the filesystem used to handle the file for `--backend_store_uri` does not support the file Lock operation (observed with the Azure File NFS solution).

To confirm whether you're running in Azure cloud hit `apolo admin get-clusters`.

A work-around for this is to use a platform `disk:` to host the SQLite data, or to use a dedicated SQL DB, for instance, [PostgreSQL hosted on the platform](https://github.com/apolo-actions/postgres).
