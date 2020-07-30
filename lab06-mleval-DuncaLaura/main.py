from math import sqrt

from predictionPerformance import regression_prediction, multi_target_regression, binary_labels_classification, \
    binary_probabilities_classification, multi_class_classification
from lossFunctions import huber_loss
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_absolute_error, \
    mean_squared_error


def print_mean_errors(mae, rmse):
    print("Mean Absolute Error : {},  Root Mean Square Error : {}".format(mae, rmse))


def print_acc_prec_recall(accuracy, precision, recall):
    print("acuratete : {}, precizie : {}, rapel : {}".format(accuracy, precision, recall))


def ui():
    while True:
        # print("\n" * get_terminal_size().lines, end='')
        print("\n1 - performanta predictiei unei regresii single-target")
        print("2 - performanta unei clasificari binare cu outputuri de tip eticheta")
        print("3 - performanta unei clasificari binare cu outputuri de tip probabilitati")
        print("4 - eroarea de predictie a unei regresii multi-target")
        print("5 - eroarea de predictie a unei clasificari multi-clasa")
        print("6 - loss-ul in cazul unei regresii")
        print("7 - loss-ul in cazul unei clasificari binare")
        print("8 - loss-ul in cazul unei clasificari multi-class")
        print("9 - loss-ul in cazul unei clasificari multi-label")
        print("0 - iesire din meniu")
        cmd = input("Comanda : ")
        cmd.split()
        if cmd == "0":
            break
        if cmd == "1":
            realOutputs = [10, 8.9, 17.4, 13.2, 29.7, 12.2, 17.1, 19.8, 20, 16.8]
            computedOutputs = [17, 10, 12, 11.9, 25.7, 12.2, 8.8, 10, 24.5, 19]
            mae, rmse = regression_prediction(realOutputs, computedOutputs)
            print_mean_errors(mae, rmse)

        if cmd == "2":
            print("Setul de date echilibrat : ")
            realLabels = ["depressed", "healthy", "depressed", "healthy", "depressed", "depressed", "healthy",
                          "healthy"]
            computedLabels = ["depressed", "healthy", "healthy", "depressed", "healthy", "healthy", "depressed",
                              "healthy"]
            labelNames = {'positive': "depressed", 'negative': "healthy"}
            accuracy, precision, recall = binary_labels_classification(realLabels, computedLabels, labelNames)
            print_acc_prec_recall(accuracy, precision, recall)
            print("\nSetul de date neechilibrat : ")
            realLabels = ["healthy", "healthy", "depressed", "healthy", "healthy", "healthy", "depressed", "healthy"]
            computedLabels = ["depressed", "healthy", "healthy", "depressed", "healthy", "healthy", "depressed",
                              "healthy"]
            labelNames = {'positive': "depressed", 'negative': "healthy"}
            accuracy, precision, recall = binary_labels_classification(realLabels, computedLabels, labelNames)
            print_acc_prec_recall(accuracy, precision, recall)

        if cmd == "3":
            realLabels = ["depressed", "healthy", "depressed", "depressed", "depressed", "healthy", "depressed",
                          "depressed"]
            computedOutputs = [[0.7, 0.3], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1], [0.3, 0.7],
                               [0.4, 0.6]]
            accuracy, precision, recall = binary_probabilities_classification(realLabels, computedOutputs)
            print_acc_prec_recall(accuracy, precision, recall)

        if cmd == "4":
            realOutputs = [[10, 8.9, 17.4, 13.2, 29.7, 12.2, 17.1, 19.8, 20, 16.8],
                           [12, 11.2, 17, 6.9, 12.2, 12.5, 19.6, 22, 21, 15],
                           [7, 8.9, 17, 24, 29.7, 8, 17.1, 19.8, 7.9, 26],
                           [7, 9.4, 15.9, 11.1, 24, 17.9, 17, 15.9, 22.4, 11],
                           [9.9, 8.7, 17.4, 28, 29, 10, 23.9, 19.8, 8, 14]]
            computedOutputs = [[20, 8.9, 17.4, 13.2, 29.7, 12.2, 17.1, 19.8, 20, 16.8],
                               [12, 11.2, 17, 6.9, 12.2, 12.5, 19.6, 22, 21, 15],
                               [7, 8.9, 17, 24, 29.7, 8, 17.1, 19.8, 7.9, 26],
                               [7, 9.4, 15.9, 11.1, 24, 17.9, 17, 15.9, 22.4, 11],
                               [9.9, 8.7, 17.4, 28, 29, 10, 23.9, 19.8, 8, 14]]
            mae, rmse = multi_target_regression(realOutputs, computedOutputs)
            print_mean_errors(mae, rmse)
            print("Folosind Scikit-learn:")
            print_mean_errors(mean_absolute_error(realOutputs, computedOutputs), sqrt(mean_squared_error(realOutputs, computedOutputs)))

        if cmd == "5":
            realLabels = ["pasta", "soup", "rice", "rice", "fish", "fish", "soup", "soup"]
            # computedOutputs = [[0.1, 0.3, 0.5, 0.1], [0.5, 0.2, 0.2, 0.1], [0.3, 0.2, 0.2, 0.3], [0.2, 0.2, 0.4, 0.2],
            # [0.1, 0.5, 0.5, 0.3], [0.3, 0.3, 0.2, 0.2], [0.1, 0.1, 0.7, 0.1], [0.1, 0.2, 0.1, 0.6]]
            computedLabels = ["pasta", "pasta", "rice", "soup", "soup", "fish", "rice", "rice"]
            accuracy, precision, recall = multi_class_classification(realLabels, computedLabels)
            print_acc_prec_recall(accuracy, precision, recall)
            print("Folosind Scikit-learn:")
            acc = accuracy_score(realLabels, computedLabels)
            precision = precision_score(realLabels, computedLabels, average=None, labels=["fish", "rice", "soup", "pasta"])
            recall = recall_score(realLabels, computedLabels, average=None, labels=["fish", "rice", "soup", "pasta"])
            print_acc_prec_recall(acc, precision, recall)

        if cmd == "6":
            print("\nRegresie single-target: ")
            realOutputs = [0, 18.9, 17.4, 13.2, 29.7, 15, 17.1, 19.8, 20, 25]
            computedOutputs = [10, 18.9, 17.4, 13.2, 29.7, 15, 17.1, 19.8, 20, 25]
            loss = huber_loss(realOutputs, computedOutputs, 1)
            print("Huber loss : {}".format(loss))

        if cmd == "7":
            pass

        if cmd == "8":
            pass

        if cmd == "9":
            pass


ui()
