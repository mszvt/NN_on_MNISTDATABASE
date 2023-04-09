%USEFUL PLOTS

epochs = [300, 200, 150, 100, 50, 25, 10];

accuracy = [0.9768, 0.9778, 0.9785, 0.9775, 0.9771, 0.9742, 0.9632,];

misclassification_rate = [0.0222, 0.0215,0.0225,0.0229, 0.0258, 0.0368];
micro_F1 = [0.9848, 0.98623, 0.98604, 0.98548, 0.98548, 0.98325, 0.97608];


f1 = figure;
plot1 = nexttile;
plot(plot1,epochs,accuracy,'-')
xlabel(plot1,'Epochs')
ylim([0.9632 0.98])
ylabel(plot1,'Accuracy')
plot2 = nexttile;
plot(plot2,epochs, micro_F1,'-')
xlabel(plot2,'Epochs')
ylim([0.97608 0.99])
ylabel(plot2,'Micro-F1')
title(plot1, 'learning rate = 0.1, nodes 1 = 80, nodes 2 = 40, batch size = 10')
saveas(f1,'plot1.png')







