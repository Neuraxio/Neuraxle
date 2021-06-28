<!-- Thank you for opening a pull request!

Please add example code and a good description for us to properly review your code.

If this is your first contribution, you'll need to sign the Contributor License Agreement to allow us to use your changes: 
https://docs.google.com/forms/d/e/1FAIpQLSfDP3eCQoV0tMq296OfbOpNn-QkHwfJQLkS0MVjSHiZQXPw2Q/viewform

Please fill in the informations below if this is pertinent for your Pull Request. -->


# What it is

My pull request does: 


## How it works

I coded it this way: 


## Example usage

Here is how you can use this new code as a end user: 

```python 
# Note: Please make dimensions and types clear to the reader. 
# E.g.: in the event fictious data is processed in this code example.
# Replace the current code example with your own. 
# You may then use this PR code example to further document your code as a docstring!

this: Example = is_a_code_example
pass
```

__________________


## Checklist before merging PR. 

Things to check each time you contribute:
<!-- Note: you may delete this list from your PR's description -->

- [x] If this is your first contribution to Neuraxle, please read the [guide to contributing to the Neuraxle framework](https://github.com/Neuraxio/Neuraxle/blob/master/CONTRIBUTING.md).
- [ ] Your local Git username is set to your GitHub username, and your local Git email is set to your GitHub email. This is important to avoid breaking the cla-bot and for your contributions to be linked to your profile. More info: https://github.com/settings/emails 
- [ ] Argument's dimensions and types are specified for new steps (important), with examples in docstrings when needed.
- [ ] Class names and argument / API variables are very clear: there is no possible ambiguity. They also respect the existing code style (avoid duplicating words for the same concept) and are intuitive.
- [ ] Use typing like `variable: Typing = ...` as much as possible. Also use typing for function arguments and return values like `def my_func(self, my_list: Dict[int, List[str]]) -> OrderedDict[int, str]:`. 
- [ ] Classes are documented: their behavior is explained beyond just the title of the class. You may even use the description written in your pull request above to fill some docstrings accurately.
- [ ] If a numpy array is used, it is important to remember that these arrays are a special type that must be documented accordingly, and that numpy array should not be abused. This is because Neuraxle is a library that is not only limited to transforming numpy arrays. To this effect, numpy steps should probably be located in the existing numpy python files as much as possible, and not be all over the place. The same applies to Pandas DataFrames.
- [ ] Code coverage is above 90% for the added code for the unit tests.
- [ ] The above description of the pull request in natural language was used to document the new code inside the code's docstrings so as to have complete documentation, with examples.
- [ ] Respect the Unit Testing status check
- [ ] Respect the Codacy status check
- [ ] Respect the cla-bot status check (unless the cla-bot is truly broken - please try to debug it first)
- [ ] Code files that were edited were reformatted automatically using PyCharm's `Ctrl+Alt+L` shortcut. You may have reorganized imports as well.
