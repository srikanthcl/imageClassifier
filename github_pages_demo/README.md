# GitHub Pages Demo Kit (Self Sustained)

This folder helps you publish a live demo page for this image-classifier project using GitHub Pages.

Important: GitHub Pages hosts static files (HTML/CSS/JS). It cannot run Python servers such as FastAPI or Streamlit directly.

So the live page includes:
- project story and architecture
- metrics and screenshots
- usage instructions
- links to repository files
- local run commands for dashboard/API

The actual model execution still happens locally or on a cloud backend.

## Folder contents

- [TUTORIAL_GITHUB_PAGES.md](TUTORIAL_GITHUB_PAGES.md): full step-by-step guide
- [GITHUB_PAGES_TUTORIAL.pdf](GITHUB_PAGES_TUTORIAL.pdf): shareable PDF tutorial
- [setup_for_repo.ps1](setup_for_repo.ps1): copies workflow file to correct root location
- [workflow-template/deploy-pages.yml](workflow-template/deploy-pages.yml): GitHub Actions deployment workflow
- [site/index.html](site/index.html): live page entry point
- [site/styles.css](site/styles.css): styles
- [site/script.js](site/script.js): dynamic values
- [site/.nojekyll](site/.nojekyll): disables Jekyll processing

## Quick publish steps

1. Push this repository to GitHub.
2. Run [setup_for_repo.ps1](setup_for_repo.ps1) from repo root in PowerShell.
3. Commit the created workflow in .github/workflows/deploy-pages.yml.
4. Push to main branch.
5. In GitHub: Settings -> Pages -> Source = GitHub Actions.
6. Wait for workflow completion.
7. Open your live URL:
   - https://<your-username>.github.io/<repo-name>/

For full explanation, read [TUTORIAL_GITHUB_PAGES.md](TUTORIAL_GITHUB_PAGES.md).
