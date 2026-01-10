# TypeFly Branch Migration - Completion Steps

This document provides step-by-step instructions to complete the migration from TypeFly-1.0 to TypeFly-2.0 as the default branch.

## ✅ Completed Steps

The following steps have been automated and completed:

1. **Created Archive Tag**: A git tag `archive/TypeFly-1.0` has been created locally to preserve the TypeFly-1.0 branch state
2. **Migration Documentation**: Created `BRANCH_MIGRATION.md` with comprehensive migration guide
3. **Updated README**: Added a notice about the branch migration to the main README

## 📋 Manual Steps Required

The following steps require repository admin access and must be completed manually:

### Step 1: Push the Archive Tag

Run the following command to push the archive tag to GitHub:

```bash
git push origin archive/TypeFly-1.0
```

This preserves the TypeFly-1.0 state permanently on GitHub.

### Step 2: Change the Default Branch on GitHub

1. Go to: https://github.com/typefly/TypeFly/settings/branches
2. Under "Default branch", click the switch/edit icon (⇄) next to the current default branch
3. Select `TypeFly-2.0` from the dropdown menu
4. Click "Update" to confirm
5. Confirm the change in the popup dialog

### Step 3: Update Protected Branch Rules (if applicable)

If you have protected branch rules:

1. Go to: https://github.com/typefly/TypeFly/settings/branches
2. Review existing branch protection rules
3. Apply similar rules to `TypeFly-2.0` if needed
4. Update or remove rules from the old default branch

### Step 4: Update CI/CD Workflows (if applicable)

If you have GitHub Actions workflows in `.github/workflows/`, check if any reference the old default branch name:

```bash
# Only run this if .github/workflows/ directory exists
if [ -d ".github/workflows" ]; then
  grep -r "TypeFly-1.0" .github/workflows/
fi
```

Update any references to use `TypeFly-2.0` or the dynamic `${{ github.event.repository.default_branch }}` variable.

### Step 5: Notify Team and Users

Consider:

- Posting an announcement in your communication channels
- Adding a pinned issue explaining the migration
- Updating any external documentation or wiki pages
- Updating project README badges if they reference specific branches

## 🔍 Verification Steps

After completing the manual steps, verify:

1. **Default Branch Changed**: 
   - Visit https://github.com/typefly/TypeFly and confirm TypeFly-2.0 is shown
   - New clones should get TypeFly-2.0 by default

2. **Archive Tag Exists**:
   ```bash
   git ls-remote --tags origin | grep archive/TypeFly-1.0
   ```

3. **Both Branches Accessible**:
   ```bash
   git ls-remote --heads origin | grep -E "(TypeFly-1.0|TypeFly-2.0)"
   ```

## 🔄 Rollback Instructions

If you need to revert the change:

1. Go to GitHub Settings → Branches
2. Change the default branch back to `TypeFly-1.0`
3. The archive tag and documentation remain for future reference

## 📚 Key Differences Between Branches

**TypeFly-1.0**:
- Basic structure with LICENSE.md
- Earlier version of the project

**TypeFly-2.0**:
- Full implementation with LLM controller, vision skills, web UI
- Contains all the major features and improvements

## ❓ Troubleshooting

**Q: Users getting errors after switching default branch?**
A: Users with existing clones should run:
```bash
git fetch origin
git checkout TypeFly-2.0
git pull origin TypeFly-2.0
```

**Q: Can we delete TypeFly-1.0 branch?**
A: Not recommended. Keep it for historical reference and users who may still need it. The archive tag ensures it's preserved even if the branch is eventually deleted.

**Q: What happens to existing PRs?**
A: Existing PRs targeting the old default branch will need to be retargeted to TypeFly-2.0 or kept as-is if appropriate.

---

**Created**: 2026-01-10
**Status**: Awaiting manual completion of Steps 1-5
