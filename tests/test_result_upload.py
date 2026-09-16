"""Execute generated pod commands in POSIX sh with local stand-ins for services."""

import asyncio
import os
import subprocess

import pytest

from coding_agent_bench.job import OpenshiftJob


SHELL_STUBS = r'''
record() {
    printf '%s\n' "$1" >> "$TRACE"
    if [ "$FAIL_STAGE" = "$1" ]; then return 23; fi
}
before() { record before; }
python3() {
    case "$2" in
        *new_host*) record url ;;
        *) record parent ;;
    esac
}
uv() {
    shift 3
    case "$1" in
        harbor)
            record harbor
            return "$HARBOR_RC"
            ;;
        aws)
            [ "$AWS_ACCESS_KEY_ID" = "$MINIO_ROOT_USER" ] || return 99
            [ "$AWS_SECRET_ACCESS_KEY" = "$MINIO_ROOT_PASSWORD" ] || return 99
            [ "$AWS_DEFAULT_REGION" = us-east-1 ] || return 99
            [ "$AWS_EC2_METADATA_DISABLED" = true ] || return 99
            shift 3
            case "$1 $2" in
                's3api head-bucket') record head ;;
                's3 mb') record bucket ;;
                's3 rm') record remove ;;
                's3 cp')
                    case "$4" in
                        s3://*) record download ;;
                        *) record upload ;;
                    esac
                    ;;
                *) return 99 ;;
            esac
            ;;
        *) return 99 ;;
    esac
}
'''


def run_shell(tmp_path, command, harbor_rc=0, fail_stage=""):
    trace = tmp_path / "trace"
    result = subprocess.run(
        ["sh", "-c", SHELL_STUBS + command],
        env={
            **os.environ,
            "TRACE": str(trace),
            "HARBOR_RC": str(harbor_rc),
            "FAIL_STAGE": fail_stage,
            "MINIO_ROOT_USER": "test user",
            "MINIO_ROOT_PASSWORD": "test password with spaces",
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 99, result.stderr
    return result.returncode, trace.read_text().splitlines()


@pytest.fixture
def queue_api(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_STORE_PATH", str(tmp_path / "initial.db"))
    from coding_agent_bench import api

    monkeypatch.setattr(api, "job_store", api.JobStore(tmp_path / "jobs.db"))
    monkeypatch.setattr(api, "_job_queue", [])
    monkeypatch.setattr(api, "_job_event", asyncio.Event())
    return api


def enqueue_resume(api, server_url="https://old.models.example.com", new_url=None):
    api.job_store.insert(
        "original", "job with spaces", "codex", "dataset", "model", server_url, []
    )
    api.job_store.update_status("original", api.JobStatus.FAILED)
    asyncio.run(
        api.resume_job("original", api.ResumeJobRequest(server_url=new_url))
    )
    return api._job_queue[-1]


@pytest.mark.parametrize("harbor_rc", [0, 1, 143])
@pytest.mark.parametrize("resume", [False, True])
def test_upload_runs_after_harbor_and_preserves_exit_status(
    tmp_path, queue_api, harbor_rc, resume
):
    if resume:
        command = enqueue_resume(queue_api).command[2]
        expected = ["download", "parent", "harbor", "remove", "upload"]
    else:
        spec = OpenshiftJob("test")._job_spec(["harbor", "run"])
        command = spec["spec"]["template"]["spec"]["containers"][0]["args"][0]
        expected = ["harbor", "head", "upload"]

    status, calls = run_shell(tmp_path, command, harbor_rc=harbor_rc)

    assert calls == expected
    assert status == harbor_rc


@pytest.mark.parametrize("harbor_rc", [0, 1])
@pytest.mark.parametrize("resume", [False, True])
def test_upload_failure_is_not_masked(tmp_path, queue_api, harbor_rc, resume):
    if resume:
        command = enqueue_resume(queue_api).command[2]
    else:
        spec = OpenshiftJob("test")._job_spec(["harbor", "run"])
        command = spec["spec"]["template"]["spec"]["containers"][0]["args"][0]

    status, calls = run_shell(
        tmp_path, command, harbor_rc=harbor_rc, fail_stage="upload"
    )

    assert calls[-1] == "upload"
    assert status == 23


@pytest.mark.parametrize("failure", ["download", "parent", "url"])
def test_failed_resume_setup_does_not_remove_remote_results(
    tmp_path, queue_api, failure
):
    command = enqueue_resume(
        queue_api, new_url="https://new.models.example.com"
    ).command[2]

    status, calls = run_shell(tmp_path, command, fail_stage=failure)

    assert status == 23
    assert calls[-1] == failure
    assert "harbor" not in calls
    assert "remove" not in calls
    assert "upload" not in calls


def test_resume_removal_failure_stops_upload(tmp_path, queue_api):
    command = enqueue_resume(queue_api).command[2]

    status, calls = run_shell(tmp_path, command, fail_stage="remove")

    assert status == 23
    assert calls == ["download", "parent", "harbor", "remove"]


def test_failed_before_script_stops_job(tmp_path):
    spec = OpenshiftJob("test")._job_spec(["harbor", "run"], ["before"])
    command = spec["spec"]["template"]["spec"]["containers"][0]["args"][0]

    status, calls = run_shell(tmp_path, command, fail_stage="before")

    assert status == 23
    assert calls == ["before"]


def test_missing_bucket_is_created_before_upload(tmp_path):
    spec = OpenshiftJob("test")._job_spec(["harbor", "run"])
    command = spec["spec"]["template"]["spec"]["containers"][0]["args"][0]

    status, calls = run_shell(tmp_path, command, fail_stage="head")

    assert status == 0
    assert calls == ["harbor", "head", "bucket", "upload"]


def test_managed_resume_updates_url_after_download(tmp_path, queue_api, monkeypatch):
    class Nebius:
        async def acquire_instance(self, *_args, **_kwargs):
            return "instance", "https://new.models.example.com"

        async def mark_job_started(self, _name):
            pass

        async def mark_job_completed(self, _name):
            pass

    commands = []

    async def capture_command(_job_id, command, **_kwargs):
        commands.append(command[2])

    monkeypatch.setattr(queue_api, "_nebius", Nebius())
    monkeypatch.setattr(queue_api, "_run_job", capture_command)
    queued = enqueue_resume(queue_api, server_url="nebius-h200")

    asyncio.run(queue_api._process_queued_job(queued))
    assert len(commands) == 1
    status, calls = run_shell(tmp_path, commands[0])

    assert status == 0
    assert calls == ["download", "parent", "url", "harbor", "remove", "upload"]
