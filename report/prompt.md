You will continue the report in report/. It is a machine learning student report. 

First, you can ignore ch-sims for now in the next experiments. Leave it as it is in the report for now.

The purpose of this part is to reproduce what has been done in the research article here at paper-presentation-support/.

Then, you will have to ensure that the basic encoders be able to learn everything from the train set, (that is to say to overfit on the train split ?). If not, make the encoders heavier. This part is not shown in the report.

Once done, you will measure their solo performance on the dataset mosi only. You explain briefly how they encode the information, then you will show their performance in the report.

After that, you'll train the total concat_fusion model from zero using the basic encoders (the classifier head in the paper), tell briefly how does it work, and show the performances too in the report. Then you will interpret the results.