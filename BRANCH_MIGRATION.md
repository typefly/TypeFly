# Branch Migration: TypeFly-1.0 to TypeFly-2.0

This document outlines the process of archiving TypeFly-1.0 and migrating to TypeFly-2.0 as the default branch.

## Archive Status

✅ **TypeFly-1.0 has been archived**

- Archive tag: `archive/TypeFly-1.0`
- Tag reference: Points to commit `6f40a91` (Create LICENSE.md)
- The tag has been created locally and needs to be pushed to the remote repository

### Pushing the Archive Tag

To push the archive tag to the remote repository, run:
```bash
git push origin archive/TypeFly-1.0
```

After pushing, you can always access the TypeFly-1.0 state via:
```bash
git checkout archive/TypeFly-1.0
```

## TypeFly-2.0 Branch

TypeFly-2.0 is the next major version with significant updates:
- Latest commit: `d885c86` (update readme)
- Contains new features and improvements over TypeFly-1.0
- Full implementation of the TypeFly system with:
  - LLM controller and planner
  - Vision skills and YOLO integration
  - MiniSpec interpreter
  - Web UI for interaction
  - Docker deployment support

## Changing the Default Branch

To make TypeFly-2.0 the default branch on GitHub, follow these steps:

### Via GitHub Web Interface (Recommended)

1. Navigate to the repository: https://github.com/typefly/TypeFly
2. Go to **Settings** → **Branches** (in the left sidebar)
3. Under "Default branch", click the switch icon next to the current default branch
4. Select `TypeFly-2.0` from the dropdown
5. Click "Update" and confirm the change

### Important Notes

- **Protected Branch Rules**: If there are any protected branch rules on the old default branch, you may need to:
  - Apply similar rules to TypeFly-2.0 before making it default
  - Remove or update rules from the old default branch after the change

- **CI/CD Workflows**: Update any GitHub Actions or CI/CD workflows that reference the old default branch name

- **Documentation Links**: Update any documentation that references the default branch

- **Clone URLs**: Users who clone the repository after this change will automatically get TypeFly-2.0

## What Happens After Changing the Default Branch

1. New clones will check out TypeFly-2.0 by default
2. Pull requests will default to target TypeFly-2.0
3. The repository homepage will show TypeFly-2.0 content
4. TypeFly-1.0 remains accessible via its branch name and the archive tag

## For Repository Users

If you have an existing local clone:

```bash
# Update your local repository
git fetch origin

# Switch to the new default branch
git checkout TypeFly-2.0
git pull origin TypeFly-2.0
```

## Rollback (if needed)

If you need to revert to TypeFly-1.0 as the default:

1. Follow the same steps in GitHub Settings
2. Select `TypeFly-1.0` as the default branch
3. The archive tag will remain for historical reference

---

**Date of Migration**: 2026-01-10
**Performed by**: GitHub Copilot
