# Contributing to Neuraxle


## First steps

For contributing, first, read the README.

We'd love to see you comment in an issue if you want to work on it.

You can as well suggest new features by creating new issues. Don't hesitate to bring new ideas.


## Before coding

New contributor? Follow this checklist to get started right on track:

- [ ] Your local Git username is set to your GitHub username, and your local Git email is set to your [GitHub email](https://github.com/settings/emails). This is important to avoid breaking the cla-bot and for your contributions to be linked to your profile. If at least 1 contribution is not commited properly using the good credentials, the cla-bot will break until your [re-commit it](https://stackoverflow.com/questions/20002557/how-to-remove-a-too-large-file-in-a-commit-when-my-branch-is-ahead-of-master-by/39768343#39768343).
- [ ] Use the PyCharm IDE with PyTest to test your code. Reformatting your code at every file save is a good idea, using [PyCharm's `Ctrl+Alt+L` shortcut](https://www.jetbrains.com/help/pycharm/reformat-and-rearrange-code.html). You may reorganize imports automatically as well, as long as your project root is well configured. Run the tests to see if everything works, and always ensure that all tests run before opening a pull request as well.
- [ ] We recommend letting PyCharm manage the virtual environment by [creating a new one just for this project](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#existing-environment), and (using PyTest as a test runner in PyCharm](https://www.jetbrains.com/help/pycharm/pytest.html#pytest-fixtures). This is not required, but should help in getting you started.
- [ ] Please [make your pull request(s) editable](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork), such as for us to add you to the list of contributors if you didn't add the entry, for example.
- [ ] To contribute, first fork the project, then do your changes, and then [open a pull request in the main repository](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).
- [ ] Sign the [Contributor License Agreement (CLA)](https://docs.google.com/forms/d/e/1FAIpQLSfDP3eCQoV0tMq296OfbOpNn-QkHwfJQLkS0MVjSHiZQXPw2Q/viewform) to allow Neuraxio to use and publish your contributions under the Apache 2.0 license, in order for everyone to be able to use your open-source contributions. Follow the instructions of the cla-bot upon opening the pull request.


## Pull Requests

You will then be able to open pull requests. The instructions in the [pull request template](https://www.neuraxle.org/stable/Neuraxle/.github/pull_request_template.html) will be shown to you upon creating each pull request.


## Code Reviews

We do code review. We expect most of what we suggest to be fixed. This is a machine learning framework. This means that it is the basis for several other projects. Therefore, the code **must** be clean, understandeable (easy to read), and documented, as many people will read and use what you have coded. Please respect pep8 as much as possible, and try as much as possible to create clean code with a good Oriented Object Programming OOP design. It is normal and expected that your Pull Requests have lots of review comments.


## Reviewing other's code

We love that contributors review each other's code as well, when they have time.


## Publishing project to PyPI

**For official project maintainers only:** you may follow these instructions to upload a new version of tha package on pip:
- https://github.com/Neuraxio/Neuraxle/wiki/How-to-deploy-a-new-package-(or-version-of-package)-to-PyPI
