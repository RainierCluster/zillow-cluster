## Rainier's Cluster Project using Zillow Data Set
##### Presented by Cris Giovanoni and Misty Garcia

### Goals
- Using clustering methodologies
- Identify what is driving the difference in zestimate to sales price "logerror"
- Build a model to predict your target "logerror"

### Hypotheses
- Noteable clusters will be found based on location
- A model containing the clusters will better predict logerror

### Components
- Clusters
- Teach others
- Use statistical tests
- Visualize to explore
- Scale
- Impute
- Feature engineering
- Code resuseability/modularity
- Stages of pipeline

### Deliverables
- Notebook (reduce_zillow_logerror.ipynb)
- Supporting materials
- Present findings/learnings to class

### Required to run
- Access to the bayes sql database
- env.py file with host, username, and password

### How to reproduce
- Pull all .py files
    - acquire.py
    - prep.py
    - split_scale.py
    - explore.py
    - model.py
- Add personal env.py file
- Run through reduce_zillow_logerror.ipynb pipeline
    - Random states of 123 are set in .py files

