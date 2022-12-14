# Contributing to RLPack

Thank you for your interest in contributing to RLPack.

### Add a feature or fix a bug in RLPack

If you would like to add a new feature or fix a bug, [raise an issue](https://github.com/kartik2309/RLPack/issues)
in our repository on GitHub. We can further discuss the details on the issue and its subsequent implementation. Once 
the discussion is concluded, you can raise a PR which will be reviewed.


### Documentation

RLPack relies on Doxygen to build its documentation. Hence, you must make sure that you have written the doc-strings
for methods, classes and variables. You must also make sure of the following: 
    
- If you have added a new feature, you must create a markdown page in 
[docs/](https://github.com/kartik2309/RLPack/tree/master/docs) in appropriate directory describing the feature. For
example, if it's a new agent, it should be added in 
[docs/agents/]((https://github.com/kartik2309/RLPack/tree/master/docs/agents/)). You may also be required to change
[doxygen_layout.xlm](https://github.com/kartik2309/RLPack/blob/algorithm/actor-critic/docs/doxygen_layout.xlm) to
correctly display your newly introduced feature in the navigation tree. You must also update the index.md appropriately.

- If you have fixed a bug which changes any argument or introduces new classes, methods or variables, they must be 
documented correctly.

Once done, you can run the following command for project's root directory to update the doxygen pages
```zsh
cd docs/
doxygen doxygen_config.txt
```

### Writing new markdown files

When you introduce a new feature, you maybe required to a add new markdown file describing the feature. As a general 
guideline, a markdown introducing a feature must fall under either category of model or agent. This is however not a
strict guideline and can be discussed. You must update index.md for 
[docs/models/index.md]((https://github.com/kartik2309/RLPack/tree/master/docs/models/index.md)) or for
[docs/agents/index.md]((https://github.com/kartik2309/RLPack/tree/master/docs/agents/index.md)) appropriately. 

The description must contain the following: 

- Title: The title of the new feature. 
- Description: A brief description about the feature, the class it uses in rlpack and its class it inherits from (if 
valid).
- Miscellaneous: This can include a detailed description on the feature, how to use it, implementation details etc. 
- Keyword: Your new feature must be accessible via keyword, which should be shown here.

You must register your new feature in 
[register](https://github.com/kartik2309/RLPack/blob/master/rlpack/utils/base/register.py). Here you must link the 
classes correctly and add the keyword by which the new feature can be accessible. This should be the same keyword to
be displayed in markdown file for the new feature.