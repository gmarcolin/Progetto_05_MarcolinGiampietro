import os
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.sensors.base import BaseSensorOperator
import requests


class GitHubWorkflowSensor(BaseSensorOperator):
    """
    Sensore per monitorare lo stato di un workflow GitHub Actions.
    Esegue il polling in modalità 'reschedule', liberando il worker tra un controllo e l'altro.
    Verifica che status sia 'completed' e conclusion sia 'success'.
    """

    template_fields = ("workflow_name",)

    def __init__(
        self,
        workflow_name: str,
        xcom_push_key: str = "workflow_run_id",
        check_deploy_gate: bool = False,
        **kwargs,
    ):
        # Default reschedule e timeout
        kwargs.setdefault("mode", "reschedule")
        kwargs.setdefault("poke_interval", 30)
        kwargs.setdefault("timeout", 1800)  # 30 minuti max
        super().__init__(**kwargs)
        self.workflow_name = workflow_name
        self.xcom_push_key = xcom_push_key
        self.check_deploy_gate = check_deploy_gate

    def poke(self, context):
        ti = context["ti"]

        # Se è il sensore di deploy, verifica prima se il Gatekeeper ha approvato il rilascio
        if self.check_deploy_gate:
            deploy_approved = ti.xcom_pull(task_ids="gatekeep_and_decide", key="deploy_approved")
            if not deploy_approved:
                self.log.info("Deploy non approvato dal Gatekeeper. Salto il sensore di deploy.")
                raise AirflowSkipException("Deploy non approvato dal Gatekeeper.")

        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPOSITORY")
        if not token or not repo:
            raise AirflowException("GITHUB_TOKEN o GITHUB_REPOSITORY non configurati nell'ambiente.")

        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        url = f"https://api.github.com/repos/{repo}/actions/runs"

        try:
            res = requests.get(url, headers=headers, params={"per_page": 10}, timeout=20)
            if res.status_code != 200:
                self.log.warning(f"Chiamata API GitHub fallita ({res.status_code}): {res.text}")
                return False

            runs = res.json().get("workflow_runs", [])
            target_run = None
            for r in runs:
                if self.workflow_name.lower() in r.get("name", "").lower():
                    target_run = r
                    break

            if not target_run:
                self.log.info(f"Nessuna run trovata per '{self.workflow_name}'. In attesa...")
                return False

            status = target_run.get("status")
            conclusion = target_run.get("conclusion")
            run_id = target_run.get("id")

            self.log.info(
                f"Workflow '{self.workflow_name}' (ID: {run_id}) -> Status: {status} | Conclusion: {conclusion}"
            )

            # Se ancora in corso, cede il worker e riprova al prossimo poke
            if status != "completed":
                return False

            # Se completato, verifichiamo la conclusion
            if conclusion == "success":
                self.log.info(f"Workflow '{self.workflow_name}' completato con SUCCESS!")
                ti.xcom_push(key=self.xcom_push_key, value=run_id)
                return True
            else:
                # Fail-fast immediato: errore esplicito senza attendere il timeout
                raise AirflowException(
                    f"Workflow GitHub '{self.workflow_name}' (ID: {run_id}) FALLITO con conclusion='{conclusion}'."
                )

        except (AirflowException, AirflowSkipException):
            raise
        except Exception as e:
            self.log.warning(f"Errore temporaneo durante il poke: {e}")
            return False
