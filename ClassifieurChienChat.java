
// ClassifieurChienChat.java
import java.io.File;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

public class ClassifieurChienChat {
    public static final float MSE_LIMITE = 0.04f;
    public static final int MAX_EPOQUES = 500; // Mécanisme de sécurité pour le rapport

    public static void main(String[] args) {
        String dossierTrain = "dataset_groupe_6/train"; // À adapter si besoin
        String dossierTest = "dataset_groupe_6/test";

        System.out.println("Chargement du dataset...");
        // CORRECTION : Utilisation de dossierTest et désactivation du miroir (false)
        // pour le jeu de test
        List<Image> trainImages = chargerDataset(dossierTrain, Image.ModeCouleur.GRIS, true);
        List<Image> testImages = chargerDataset(dossierTest, Image.ModeCouleur.GRIS, false);

        if (trainImages.isEmpty()) {
            System.out.println("Aucune image trouvée.");
            return;
        }

        melangerFisherYates(trainImages); // Biais d'ordre cassé

        int nbPixels = trainImages.get(0).donnees().length;
        Neurone modele = new NeuroneSigmoide(nbPixels);

        System.out.println("Entraînement en cours (Max " + MAX_EPOQUES + " époques)...");
        entrainerPlafonne(modele, trainImages);

        System.out.println("\nÉvaluation sur le jeu de test...");
        evaluer(modele, testImages);
    }

    // CORRECTION : Remplacement de Image.Mode par Image.ModeCouleur + suppression
    // de la méthode doublon vide
    public static List<Image> chargerDataset(String repertoire, Image.ModeCouleur mode, boolean appliquerMiroir) {
        List<Image> images = new ArrayList<>();
        try {
            List<String> paths = Files.walk(Paths.get(repertoire))
                    .filter(Files::isRegularFile)
                    .map(Path::toString)
                    .filter(p -> p.endsWith(".jpg") || p.endsWith(".png"))
                    .collect(Collectors.toList());

            for (String p : paths) {
                images.add(new Image(p, mode, false));
                if (appliquerMiroir)
                    images.add(new Image(p, mode, true)); // Augmentation de données
            }
        } catch (Exception e) {
            System.err.println("Erreur lors du chargement du dossier : " + repertoire);
        }
        return images;
    }

    public static void melangerFisherYates(List<Image> liste) {
        Random rnd = new Random();
        for (int i = liste.size() - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            Image a = liste.get(index);
            liste.set(index, liste.get(i));
            liste.set(i, a);
        }
    }

    public static int entrainerPlafonne(Neurone n, List<Image> train) {
        int epoque = 0;
        double mse = 0;
        float[][] entrees = new float[train.size()][];
        float[] labels = new float[train.size()];

        // CORRECTION : Boucle d'initialisation pour copier proprement TOUTES les images
        // du dataset
        for (int i = 0; i < train.size(); i++) {
            // Associe la cible (1.0f si c'est un chat, aligné avec la logique d'évaluation)
            labels[i] = (train.get(i).label() == 1) ? 1.0f : 0.0f;

            // --- DEBUT DU TEST SANS NORMALISATION ---
            int[] donnéesBrutes = train.get(i).donnees();
            entrees[i] = new float[donnéesBrutes.length];
            for (int k = 0; k < donnéesBrutes.length; k++) {
                entrees[i][k] = (float) donnéesBrutes[k];
            }
            // ----------------------------------------
            // NOTE SCIENTIFIQUE : Pour repasser au mode normalisé standard, commente le
            // bloc ci-dessus et mets :
            // entrees[i] = train.get(i).obtenirDonneesNormalisees();
        }

        // Boucle d'apprentissage saine (sans conflit sur la variable locale 'i')
        do {
            mse = 0;
            for (int i = 0; i < entrees.length; ++i) {
                n.metAJour(entrees[i]);
                float delta = labels[i] - n.sortie();
                mse += delta * delta;
                for (int j = 0; j < entrees[i].length; ++j) {
                    n.synapses()[j] += entrees[i][j] * 0.0001f * delta;
                }
                n.fixeBiais(n.biais() + 0.0001f * delta);
            }
            mse /= entrees.length;
            epoque++;
        } while (mse > MSE_LIMITE && epoque < MAX_EPOQUES);

        System.out.printf("Entraînement terminé: %d époques, MSE finale = %.4f\n", epoque, mse);
        return epoque;
    }

    public static float[] evaluer(Neurone n, List<Image> test) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (Image img : test) {

            // CORRECTION : Conversion obligatoire de int[] vers float[] pour correspondre à
            // la signature de metAJour()
            int[] donnéesBrutes = img.donnees();
            float[] entreesTest = new float[donnéesBrutes.length];
            for (int k = 0; k < donnéesBrutes.length; k++) {
                entreesTest[k] = (float) donnéesBrutes[k];
            }
            // NOTE : Si tu passes en mode normalisé dans l'entraînement, utilise ici :
            // float[] entreesTest = img.obtenirDonneesNormalisees();

            n.metAJour(entreesTest);
            boolean predictChat = n.sortie() >= 0.5f;
            boolean estChat = img.label() == 1;

            if (predictChat && estChat)
                tp++;
            else if (!predictChat && !estChat)
                tn++;
            else if (predictChat && !estChat)
                fp++;
            else
                fn++;
        }

        float accuracy = (float) (tp + tn) / test.size();
        float precision = (tp + fp) == 0 ? 0 : (float) tp / (tp + fp);
        float recall = (tp + fn) == 0 ? 0 : (float) tp / (tp + fn);
        float f1 = (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);

        System.out.println("Matrice de Confusion :");
        System.out.printf("TP: %d | FP: %d\n", tp, fp);
        System.out.printf("FN: %d | TN: %d\n", fn, tn);
        System.out.printf("Précision: %.2f | Rappel: %.2f | Score F1: %.2f | Taux Réussite: %.2f%%\n",
                precision, recall, f1, accuracy * 100);

        return new float[] { accuracy, precision, recall, f1 };
    }
}