# Model Patch Notes

This file records model-side compatibility notes relevant to reproducing local inference runs in the WildBioWiki project.

## Scope

The release package does not redistribute patched third-party model source trees or model weights. During local experimentation, some open models required environment-specific or code-level compatibility edits in order to run inference reliably.

Some third-party multimodal models required local compatibility handling on the development machines used for this project.

## What To Record During Reproduction

If you reproduce or extend the released inference workflow, record any local changes you make in the following categories:

- tokenizer or processor loading fixes
- `trust_remote_code` requirements
- import-path or package-version fixes
- model forward-pass or generation wrappers
- vision tower / projector loading fixes
- precision, device-map, or attention backend workarounds

## Recommended Reproduction Practice

For each patched model, save:

- the model name and exact checkpoint identifier
- the commit hash of the third-party model repository, if applicable
- the local diff or patched file list
- the package versions used in the environment
- a short note describing why the patch was required

## Release Note

This file is intentionally conservative: it documents that local compatibility edits were part of the experimental workflow, but it does not claim that every reproduction will need the same patch set on a different machine or software stack.
