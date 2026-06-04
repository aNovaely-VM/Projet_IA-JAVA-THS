import java.io.*;
import java.nio.file.*;
import java.util.*;

/*
 * Point d'entrée principal du programme de classification d'images par réseau de neurones.
 * Gère l'entraînement (train), l'évaluation (test) et le tri automatique (sort) d'images
 * en chats, chiens ou animaux sauvages, selon plusieurs architectures et modes de traitement.
 *
 * Regarder README.md pour les instructions.
 */
public class Compilation
{
    // =========================================================
    //  VARIABLES DE CONFIGURATION
    // =========================================================

    // Chemins vers les datasets
    static final String DATASET_TRAIN = "dataset_groupe_6/train/";
    static final String DATASET_TEST  = "dataset_groupe_6/test/";
    static final String DATASET_SORT  = "dataset_groupe_6/naming/to_sort/";

    // Dossiers de sortie du tri (binaire = 1 neurone / multiple = 3 neurones)
    static final String DATASET_SORT_OUT_BINAIRE  = "dataset_groupe_6/naming/binaire/";
    static final String DATASET_SORT_OUT_MULTIPLE = "dataset_groupe_6/naming/multiple/";

    // Classes à reconnaître et leurs labels entiers associés (index parallèles)
    static final String[] CLASSES = {"cat", "dog", "wild"};
    static final int[]    LABELS  = {0,      1,     2};

    // Taille de redimensionnement des images (RESIZE x RESIZE pixels)
    static final int RESIZE = 64;

    // Options présentées à l'utilisateur dans les menus interactifs
    private static final String[] OPTIONS_COULEUR       = {"GRI", "COL", "TSL"};
    private static final String[] OPTIONS_NEURONE       = {"heavyside", "relu", "sigmoide"};
    private static final String[] OPTIONS_TRAITEMENT    = {"simple", "mirror", "histogramme (Oblige COL ou TSL)", "HOG (Oblige GRI)"};
    private static final String[] OPTIONS_ARCHITECTURE  = {"1 Neurone (Cat ou non - Binaire)", "3 Neurones (Cat, Dog, Wild - Multiclasse)"};
    private static final String[] OPTIONS_ORDRE         = {"Ordre séquentiel (Linéaire)", "Mélange aléatoire (Shuffle)", "Mélange à chaque itération (Deep Shuffle ─ très coûteux)"};
    private static final String[] OPTIONS_TYPE_LOGS     = {"Binaire (1 Neurone)", "Multiple (3 Neurones)"};
    private static final String[] OPTIONS_OUI_NON       = {"Oui", "Non"};

    // Paramètres actifs sélectionnés par l'utilisateur au runtime
    static String  typeCouleurActif      = "GRI";
    static String  typeNeuroneActif      = "sigmoide";
    static String  typeTraitementActif   = "simple";
    static boolean modeMonoNeuroneActif  = false;   // true = architecture binaire (1 neurone), false = multiclasse (3 neurones)
    static boolean modeShuffleActif      = false;   // true = images mélangées aléatoirement avant entraînement
    static boolean modeDeepShuffleActif  = false;   // true = images mélangées à chaque itération (coûteux)
    static boolean modeNormalisationActif= false;   // true = chaque vecteur d'entrée est normalisé à norme unitaire

    // Hyperparamètres du réseau de neurones
    static final float MSE_LIMITE     = 0.01f;   // Seuil MSE en dessous duquel on arrête l'entraînement
    static final float ETA            = 0.0001f; // Taux d'apprentissage (pas de descente de gradient)
    static final int   MAX_ITERATIONS = 200;     // Nombre maximum d'itérations de la boucle d'entraînement

    // Dossiers de sauvegarde des poids et de la configuration (choisis dynamiquement selon le mode)
    static final String SAVE_DIR_BINAIRE  = "Logs/binaire/";
    static final String SAVE_DIR_MULTIPLE = "Logs/multiple/";

    // Cache des vecteurs transformés : évite de recalculer HOG/miroir/etc. pour une même image.
    // Clé = chemin absolu du fichier image (String), valeur = vecteur float[].
    // Le chemin garantit l'unicité même entre deux objets Image distincts pointant vers le même fichier.
    private static final Map<String, float[]> cacheTransformations = new HashMap<>();

    // =========================================================
    //  HELPERS : chemins dynamiques selon le mode actif
    // =========================================================

    /** Retourne le dossier de logs (poids + config) selon le mode courant. */
    static String saveDir() {
        return modeMonoNeuroneActif ? SAVE_DIR_BINAIRE : SAVE_DIR_MULTIPLE;
    }

    /** Retourne le dossier de destination du tri selon le mode courant. */
    static String sortOutDir() {
        return modeMonoNeuroneActif ? DATASET_SORT_OUT_BINAIRE : DATASET_SORT_OUT_MULTIPLE;
    }

    // =========================================================
    //  MAIN
    // =========================================================
    public static void main(String[] args) throws Exception
    {
        System.out.println("┌──────────────────────────────────────────┐");
        System.out.println("│  Compilation.java ─ Réseau de neurones   │");
        System.out.println("└──────────────────────────────────────────┘\n");

        Scanner sc = new Scanner(System.in);

        // Boucle principale : reste dans le menu jusqu'au choix 7 (Quitter)
        while (true) {

            System.out.println("Que voulez-vous faire ?");
            System.out.println("  1) Entraîner puis évaluer (Train + Test)");
            System.out.println("  2) Entraîner puis trier (Train + Sort)");
            System.out.println("  3) Entraîner uniquement (Train only)");
            System.out.println("  4) Évaluer uniquement (Test only ─ ! Nécessite un modèle déjà entraîné ! )");
            System.out.println("  5) Trier uniquement (Sort only ─ ! Nécessite un modèle déjà entraîné ! )");
            System.out.println("  6) Nettoyer les sorties (Clear ─ supprime les images triées dans naming/binaire et naming/multiple)");
            System.out.println("  7) Quitter le programme");
            System.out.print("\nVotre choix [1-7] : ");
            String choix = sc.nextLine().trim();
            System.out.println();

            if (choix.equals("7")) {
                System.out.println("Au revoir !");
                break;
            }

            // Les options 1, 2 et 3 nécessitent une configuration complète avant l'entraînement
            boolean necessiteEntrainement = choix.equals("1") || choix.equals("2") || choix.equals("3");

            // --- CONFIGURATION DYNAMIQUE ---
            if (necessiteEntrainement) {
                typeNeuroneActif = menuChoix(sc, "Choisissez le type de neurone :", OPTIONS_NEURONE);

                // Le traitement peut forcer automatiquement un mode couleur (HOG → GRI, histogramme → COL)
                String choixTraitementRaw = menuChoix(sc, "Choisissez le type de traitement :", OPTIONS_TRAITEMENT);
                // Extraction du premier mot uniquement pour ignorer les parenthèses d'aide à l'affichage
                typeTraitementActif = choixTraitementRaw.contains(" ")
                    ? choixTraitementRaw.split(" ")[0].trim()
                    : choixTraitementRaw.trim();

                if (typeTraitementActif.equalsIgnoreCase("HOG")) {
                    System.out.println("[INFO] Le mode HOG requiert des niveaux de gris. Configuration automatique sur 'GRI'.\n");
                    typeCouleurActif = "GRI";
                } else if (typeTraitementActif.equalsIgnoreCase("histogramme")) {
                    System.out.println("[INFO] Le mode Histogramme requiert de la couleur. Configuration automatique sur 'COL'.\n");
                    typeCouleurActif = "COL";
                } else {
                    typeCouleurActif = menuChoix(sc, "Choisissez le mode couleur :", OPTIONS_COULEUR);
                }

                String choixArch = menuChoix(sc, "Choisissez l'architecture du réseau :", OPTIONS_ARCHITECTURE);
                modeMonoNeuroneActif = choixArch.startsWith("1");

                String choixOrdre = menuChoix(sc, "Choisissez l'ordre des images :", OPTIONS_ORDRE);
                modeDeepShuffleActif = choixOrdre.contains("Deep Shuffle");
                modeShuffleActif     = choixOrdre.contains("Shuffle") && !modeDeepShuffleActif;

                String choixNorm = menuChoix(sc, "Activer la normalisation des vecteurs d'entrée ?", OPTIONS_OUI_NON);
                modeNormalisationActif = choixNorm.equals("Oui");

                // Vidage du cache pour qu'aucun vecteur calculé avec l'ancienne configuration ne soit réutilisé
                cacheTransformations.clear();
                afficherConfig();

            } else if (choix.equals("4") || choix.equals("5")) {
                // En mode évaluation/tri seul, rechargement de la configuration depuis les logs du modèle existant
                String choixLogs = menuChoix(sc, "Quel type de modèle voulez-vous utiliser ?", OPTIONS_TYPE_LOGS);
                modeMonoNeuroneActif = choixLogs.startsWith("Binaire");

                // Vérification préalable : le fichier de configuration doit exister
                File configFile = new File(saveDir() + "config.txt");
                if (!configFile.exists()) {
                    System.err.println("╔══════════════════════════════════════════════════════════╗");
                    System.err.println("║  ERREUR : Aucune configuration trouvée dans " + saveDir());
                    System.err.println("║  Impossible de lancer cette action sans modèle entraîné.");
                    System.err.println("║  → Lancez d'abord un entraînement (options 1, 2 ou 3).  ");
                    System.err.println("╚══════════════════════════════════════════════════════════╝");
                    continue;
                }

                try {
                    chargerConfiguration();
                    System.out.println("[INFO] Logs de configuration chargés depuis : " + saveDir());
                } catch (IOException e) {
                    System.err.println("[ERREUR] Lecture du fichier de configuration échouée : " + e.getMessage());
                    System.err.println("→ Relancez un entraînement pour régénérer la configuration.");
                    continue;
                }
                afficherConfig();
            }
            // option 6 (clear) et choix invalide n'ont pas besoin de config

            // --- EXÉCUTION DE L'ACTION CHOISIE ---
            switch (choix) {
                case "1" -> { train(); test(); }
                case "2" -> { train(); sort(); }
                case "3" -> train();
                case "4" -> test();
                case "5" -> sort();
                case "6" -> clear(sc);
                default  -> System.out.println("Choix invalide.\n");
            }

            System.out.println();
        }
    }

    /**
     * Affiche un menu interactif numéroté et retourne l'option choisie.
     * Redemande tant que la saisie n'est pas un entier valide dans la plage.
     */
    private static String menuChoix(Scanner sc, String titre, String[] options) {
        while (true) {
            System.out.println("- " + titre);
            for (int i = 0; i < options.length; i++) {
                System.out.printf("  %d) %s\n", (i + 1), options[i]);
            }
            System.out.print("Votre choix : ");
            try {
                int choix = Integer.parseInt(sc.nextLine().trim());
                if (choix >= 1 && choix <= options.length) {
                    System.out.println();
                    return options[choix - 1];
                }
            } catch (NumberFormatException e) { /* saisie non numérique, on reboucle */ }
            System.out.println("Choix invalide. Réessayez.\n");
        }
    }

    /** Affiche un récapitulatif lisible de tous les paramètres actifs. */
    static void afficherConfig()
    {
        System.out.println("┌────────────────────────────────────────────────────────┐");
        System.out.println("│                 CONFIGURATION ACTIVE                   │");
        System.out.println("├────────────────────────────────────────────────────────┤");
        System.out.printf("│  Dataset train    : %-34s │\n", DATASET_TRAIN);
        System.out.printf("│  Dataset test     : %-34s │\n", DATASET_TEST);
        System.out.printf("│  Dataset à trier  : %-34s │\n", DATASET_SORT);
        System.out.printf("│  Sortie du tri    : %-34s │\n", sortOutDir());
        System.out.printf("│  Dossier logs     : %-34s │\n", saveDir());
        System.out.printf("│  Architecture     : %-34s │\n", (modeMonoNeuroneActif ? "1 Neurone (Cat / Non-Cat)" : "3 Neurones (Collaboratif)"));
        System.out.printf("│  Neurone choisi   : %-34s │\n", typeNeuroneActif);
        System.out.printf("│  Mode image       : %-34s │\n", typeCouleurActif);
        System.out.printf("│  Traitement       : %-34s │\n", typeTraitementActif);
        System.out.printf("│  Dimension d'adm. : %-34s │\n", (RESIZE + "x" + RESIZE));
        System.out.printf("│  MSE cible        : %-34s │\n", MSE_LIMITE);
        System.out.printf("│  Max Itérations   : %-34s │\n", MAX_ITERATIONS);
        System.out.printf("│  Pas d'app. (Eta) : %-34s │\n", ETA);
        System.out.printf("│  Normalisation    : %-34s │\n", (modeNormalisationActif ? "Activée" : "Désactivée"));
        System.out.printf("│  Deep Shuffle     : %-34s │\n", (modeDeepShuffleActif ? "Activé (shuffle/itération)" : "Désactivé"));
        System.out.println("└────────────────────────────────────────────────────────┘\n");
    }

    // =========================================================
    //  SAUVEGARDE ET CHARGEMENT DE LA CONFIGURATION
    // =========================================================

    /**
     * Persiste les paramètres actifs dans un fichier texte clé=valeur
     * dans le dossier de logs correspondant au mode courant.
     * Permet de retrouver exactement la même configuration lors d'un test/tri ultérieur.
     */
    private static void sauvegarderConfiguration() throws IOException {
        File dir = new File(saveDir());
        if (!dir.exists()) dir.mkdirs();

        try (PrintWriter pw = new PrintWriter(new FileWriter(saveDir() + "config.txt"))) {
            pw.println("typeCouleurActif="      + typeCouleurActif);
            pw.println("typeNeuroneActif="      + typeNeuroneActif);
            pw.println("typeTraitementActif="   + typeTraitementActif);
            pw.println("modeMonoNeuroneActif="  + modeMonoNeuroneActif);
            pw.println("modeShuffleActif="      + modeShuffleActif);
            pw.println("modeDeepShuffleActif="  + modeDeepShuffleActif);
            pw.println("modeNormalisationActif="+ modeNormalisationActif);
        }
    }

    /**
     * Relit la configuration depuis config.txt et réhydrate les variables statiques.
     * Limite le split à 2 parts pour tolérer d'éventuels "=" dans les valeurs.
     */
    private static void chargerConfiguration() throws IOException {
        File configFile = new File(saveDir() + "config.txt");
        if (!configFile.exists()) {
            throw new FileNotFoundException("Fichier de configuration absent : " + saveDir() + "config.txt");
        }

        try (BufferedReader br = new BufferedReader(new FileReader(configFile))) {
            String ligne;
            while ((ligne = br.readLine()) != null) {
                String[] parts = ligne.split("=", 2);
                if (parts.length == 2) {
                    String cle    = parts[0].trim();
                    String valeur = parts[1].trim();
                    switch (cle) {
                        case "typeCouleurActif"      -> typeCouleurActif      = valeur;
                        case "typeNeuroneActif"      -> typeNeuroneActif      = valeur;
                        case "typeTraitementActif"   -> typeTraitementActif   = valeur;
                        case "modeMonoNeuroneActif"  -> modeMonoNeuroneActif  = Boolean.parseBoolean(valeur);
                        case "modeShuffleActif"      -> modeShuffleActif      = Boolean.parseBoolean(valeur);
                        case "modeDeepShuffleActif"  -> modeDeepShuffleActif  = Boolean.parseBoolean(valeur);
                        case "modeNormalisationActif"-> modeNormalisationActif= Boolean.parseBoolean(valeur);
                    }
                }
            }
        }
    }

    // =========================================================
    //  PIPELINE DE TRAITEMENT D'IMAGES
    // =========================================================

    /**
     * Convertit une liste d'images en matrice de vecteurs d'entrée.
     * Chaque ligne correspond à une image transformée selon la configuration active.
     * Utilise un tableau jagged (float[][]) dont la taille de chaque ligne est déterminée
     * dynamiquement par la transformation appliquée.
     */
    static float[][] chargerEntrees(List<Image> images)
    {
        if (images.isEmpty()) return new float[0][0];
        float[][] entrees = new float[images.size()][];
        for (int i = 0; i < images.size(); i++) {
            entrees[i] = obtenirVecteurTransforme(images.get(i));
        }
        return entrees;
    }

    /**
     * Retourne le vecteur transformé d'une image, en le calculant une seule fois
     * puis en le mémorisant dans le cache pour éviter les recalculs coûteux.
     * Le chemin du fichier sert de clé unique, ce qui permet à deux objets Image
     * distincts pointant vers le même fichier de partager la même entrée de cache.
     */
    static float[] obtenirVecteurTransforme(Image img) {
        String cleUnique = img.chemin();

        float[] enCache = cacheTransformations.get(cleUnique);
        if (enCache != null) return enCache;

        float[] transforme = appliquerTransformations(img);
        cacheTransformations.put(cleUnique, transforme);
        return transforme;
    }

    /**
     * Applique la chaîne de traitement complète à une image :
     *   1. Conversion colorimétrique (GRI / COL / TSL)
     *   2. Traitement optionnel (miroir, égalisation d'histogramme, HOG)
     *   3. Normalisation unitaire si activée
     * Retourne un vecteur float prêt à être fourni à un neurone.
     */
    static float[] appliquerTransformations(Image img) {
        int[] pixelsBruts = img.donnees();
        float[] donneesModifiees;
        int largeur = img.largeur();
        int hauteur = img.hauteur();

        // --- Étape 1 : conversion colorimétrique ---
        switch (typeCouleurActif) {
            case "GRI" -> {
                // Niveaux de gris : luminance pondérée (coefficients Rec. 601)
                donneesModifiees = new float[pixelsBruts.length];
                for (int i = 0; i < pixelsBruts.length; i++) {
                    if (pixelsBruts[i] > 255 || pixelsBruts[i] < 0) {
                        // Pixel encodé en RGB packed (int 0xRRGGBB)
                        int r = (pixelsBruts[i] >> 16) & 0xFF;
                        int g = (pixelsBruts[i] >> 8)  & 0xFF;
                        int b =  pixelsBruts[i]         & 0xFF;
                        donneesModifiees[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.f;
                    } else {
                        // Pixel déjà en niveaux de gris [0..255]
                        donneesModifiees[i] = pixelsBruts[i] / 255.f;
                    }
                }
            }
            case "TSL" -> donneesModifiees = convertRGBtoTSL(pixelsBruts, img.estEnNiveauxDeGris());
            default -> {
                // Mode COL : séparation de R, G, B en trois composantes consécutives dans le vecteur
                donneesModifiees = new float[pixelsBruts.length * (img.estEnNiveauxDeGris() ? 1 : 3)];
                if (img.estEnNiveauxDeGris()) {
                    for (int i = 0; i < pixelsBruts.length; i++) {
                        donneesModifiees[i] = pixelsBruts[i] / 255.f;
                    }
                } else {
                    int index = 0;
                    for (int i = 0; i < pixelsBruts.length; i++) {
                        int p = pixelsBruts[i];
                        donneesModifiees[index++] = ((p >> 16) & 0xFF) / 255.f; // Rouge
                        donneesModifiees[index++] = ((p >> 8)  & 0xFF) / 255.f; // Vert
                        donneesModifiees[index++] = ( p        & 0xFF) / 255.f; // Bleu
                    }
                }
            }
        }

        // Les modes COL et TSL produisent des vecteurs à 3 composantes par pixel
        boolean estMulticomposante = typeCouleurActif.equals("COL") || typeCouleurActif.equals("TSL");

        // --- Étape 2 : traitement optionnel ---
        switch (typeTraitementActif.toLowerCase()) {
            case "mirror"      -> donneesModifiees = appliquerEffetMiroir(donneesModifiees, largeur, hauteur, estMulticomposante);
            case "histogramme" -> donneesModifiees = appliquerEgalisationHistogramme(donneesModifiees, estMulticomposante);
            case "hog"         -> donneesModifiees = calculerHOG(donneesModifiees, largeur, hauteur, estMulticomposante);
            // "simple" : aucun traitement supplémentaire
        }

        // --- Étape 3 : normalisation L2 optionnelle ---
        if (modeNormalisationActif) {
            donneesModifiees = normaliserVecteur(donneesModifiees);
        }

        return donneesModifiees;
    }

    /**
     * Convertit un tableau de pixels RGB packed en vecteur TSL (Teinte, Saturation, Luminosité).
     * Les images en niveaux de gris reçoivent T=0, S=0, L=valeur.
     * Chaque pixel produit 3 composantes consécutives dans le tableau résultat.
     */
    private static float[] convertRGBtoTSL(int[] rbgColor, boolean estGris) {
        if (estGris) {
            float[] tsl = new float[rbgColor.length * 3];
            for (int i = 0; i < rbgColor.length; i++) {
                float val = rbgColor[i] / 255.f;
                tsl[3 * i]     = 0.0f; // Teinte indéfinie pour le gris
                tsl[3 * i + 1] = 0.0f; // Saturation nulle
                tsl[3 * i + 2] = val;  // Luminosité
            }
            return tsl;
        }

        float[] tsl = new float[rbgColor.length * 3];
        for (int i = 0; i < rbgColor.length; i++) {
            int p   = rbgColor[i];
            float r = ((p >> 16) & 0xFF) / 255.f;
            float g = ((p >> 8)  & 0xFF) / 255.f;
            float b = ( p        & 0xFF) / 255.f;

            float max = Math.max(r, Math.max(g, b));
            float min = Math.min(r, Math.min(g, b));
            float l   = (max + min) / 2.0f;
            float h   = 0.0f, s = 0.0f;

            if (max != min) {
                float d = max - min;
                // Saturation dépend du côté sombre ou clair du spectre (formule HLS standard)
                s = l > 0.5f ? d / (2.0f - max - min) : d / (max + min);

                // Calcul de la teinte selon la composante dominante
                if      (max == r) h = (g - b) / d + (g < b ? 6.0f : 0.0f);
                else if (max == g) h = (b - r) / d + 2.0f;
                else               h = (r - g) / d + 4.0f;
                h /= 6.0f; // Normalisation dans [0, 1]
            }

            tsl[3 * i]     = h;
            tsl[3 * i + 1] = s;
            tsl[3 * i + 2] = l;
        }
        return tsl;
    }

    /**
     * Retourne une copie du vecteur image où chaque ligne de pixels est retournée
     * horizontalement (symétrie axe vertical). Utile comme augmentation de données.
     * Gère les modes 1 composante (GRI) et 3 composantes (COL/TSL) via le paramètre `pas`.
     */
    private static float[] appliquerEffetMiroir(float[] src, int large, int haut, boolean multicomposante) {
        float[] dest = new float[src.length];
        int pas = multicomposante ? 3 : 1; // nombre de valeurs float par pixel

        for (int y = 0; y < haut; y++) {
            for (int x = 0; x < large; x++) {
                int idxSrc  = (y * large + x)                * pas;
                int idxDest = (y * large + (large - 1 - x))  * pas; // pixel symétrique
                for (int c = 0; c < pas; c++) {
                    dest[idxDest + c] = src[idxSrc + c];
                }
            }
        }
        return dest;
    }

    /**
     * Applique une égalisation d'histogramme pour améliorer le contraste de l'image.
     * En mode niveaux de gris : redistribue les intensités de façon uniforme sur [0,1].
     * En mode couleur (COL/TSL) : n'égalise que la luminosité pour préserver les teintes.
     */
    private static float[] appliquerEgalisationHistogramme(float[] src, boolean multicomposante) {
        float[] dest = new float[src.length];
        int bins = 256;
        int[] histogramme = new int[bins];

        if (!multicomposante) {
            // --- Mode niveaux de gris ---
            // Construction de l'histogramme des intensités
            for (float val : src) {
                int bin = Math.min(bins - 1, Math.max(0, (int)(val * 255)));
                histogramme[bin]++;
            }

            // CDF (fonction de distribution cumulative) pour le remappage
            int[] cdf = new int[bins];
            cdf[0] = histogramme[0];
            for (int i = 1; i < bins; i++) cdf[i] = cdf[i - 1] + histogramme[i];

            // Valeur minimale non nulle de la CDF (normalisation de l'égalisation)
            int cdfMin = 0;
            for (int i = 0; i < bins; i++) {
                if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
            }

            // Remappage de chaque pixel selon la CDF normalisée
            for (int i = 0; i < src.length; i++) {
                int bin = Math.min(bins - 1, Math.max(0, (int)(src[i] * 255)));
                dest[i] = ((float)(cdf[bin] - cdfMin) / (src.length - cdfMin));
            }

        } else {
            // --- Mode couleur (COL ou TSL) ---
            // Copie du tableau source ; seul le canal de luminosité sera modifié
            System.arraycopy(src, 0, dest, 0, src.length);
            int totalPixels = src.length / 3;
            boolean estTSL  = typeCouleurActif.equals("TSL");

            // Construction de l'histogramme sur la luminosité uniquement
            for (int i = 0; i < totalPixels; i++) {
                float v = estTSL
                    ? src[3 * i + 2]                                // composante L en TSL
                    : (src[3 * i] + src[3 * i + 1] + src[3 * i + 2]) / 3.0f; // luminosité moyenne en COL
                int bin = Math.min(bins - 1, Math.max(0, (int)(v * 255)));
                histogramme[bin]++;
            }

            int[] cdf = new int[bins];
            cdf[0] = histogramme[0];
            for (int i = 1; i < bins; i++) cdf[i] = cdf[i - 1] + histogramme[i];

            int cdfMin = 0;
            for (int i = 0; i < bins; i++) {
                if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
            }

            for (int i = 0; i < totalPixels; i++) {
                float vOriginale = estTSL
                    ? src[3 * i + 2]
                    : (src[3 * i] + src[3 * i + 1] + src[3 * i + 2]) / 3.0f;

                int bin = Math.min(bins - 1, Math.max(0, (int)(vOriginale * 255)));
                float vEgalisee = ((float)(cdf[bin] - cdfMin) / (totalPixels - cdfMin));

                if (estTSL) {
                    // En TSL : remplacement direct du canal L (index 2)
                    dest[3 * i + 2] = vEgalisee;
                } else {
                    // En COL : facteur proportionnel appliqué à chaque canal RGB
                    // pour conserver les ratios de couleur tout en rehaussant la luminosité.
                    // Le +0.001 évite la division par zéro sur les pixels noirs.
                    float facteur = vEgalisee / (vOriginale + 0.001f);
                    dest[3 * i]     = Math.min(1.0f, src[3 * i]     * facteur);
                    dest[3 * i + 1] = Math.min(1.0f, src[3 * i + 1] * facteur);
                    dest[3 * i + 2] = Math.min(1.0f, src[3 * i + 2] * facteur);
                }
            }
        }
        return dest;
    }

    /**
     * Calcule le descripteur HOG (Histogram of Oriented Gradients) de l'image.
     * Découpe l'image en cellules de 8x8 pixels, calcule l'orientation et la magnitude
     * du gradient dans chaque cellule, puis les accumule dans 9 bins angulaires.
     * Retourne un vecteur de taille (largeur/8) * (hauteur/8) * 9.
     * Les bordures (1 pixel) ne contribuent pas aux gradients (valeurs initialisées à 0).
     */
    private static float[] calculerHOG(float[] src, int large, int haut, boolean multicomposante) {
        // Extraction de la carte d'intensité (passage en niveaux de gris si nécessaire)
        float[][] intensite = new float[haut][large];
        int pas = multicomposante ? 3 : 1;
        for (int y = 0; y < haut; y++) {
            for (int x = 0; x < large; x++) {
                int idx = (y * large + x) * pas;
                intensite[y][x] = multicomposante
                    ? 0.299f * src[idx] + 0.587f * src[idx + 1] + 0.114f * src[idx + 2]
                    : src[idx];
            }
        }

        // Calcul des gradients par différences finies et conversion en magnitude/angle
        float[][] magnitude   = new float[haut][large];
        float[][] orientation = new float[haut][large];

        for (int y = 1; y < haut - 1; y++) {
            for (int x = 1; x < large - 1; x++) {
                float gx = intensite[y][x + 1] - intensite[y][x - 1]; // gradient horizontal
                float gy = intensite[y + 1][x] - intensite[y - 1][x]; // gradient vertical
                magnitude[y][x]   = (float) Math.sqrt(gx * gx + gy * gy);
                float angle = (float) Math.toDegrees(Math.atan2(gy, gx));
                if (angle < 0) angle += 180.0f; // normalisation dans [0°, 180°] (gradients non orientés)
                orientation[y][x] = angle;
            }
        }

        // Accumulation des magnitudes dans les histogrammes de cellules 8x8
        int tailleCellule = 8;
        int numBins       = 9;   // 9 bins couvrent [0°, 180°] par tranches de 20°
        int cellulesX     = large / tailleCellule;
        int cellulesY     = haut  / tailleCellule;
        float[] vecteurHOG = new float[cellulesX * cellulesY * numBins];

        int indexVecteur = 0;
        for (int cy = 0; cy < cellulesY; cy++) {
            for (int cx = 0; cx < cellulesX; cx++) {
                float[] histCellule = new float[numBins];
                for (int y = 0; y < tailleCellule; y++) {
                    for (int x = 0; x < tailleCellule; x++) {
                        int px  = cx * tailleCellule + x;
                        int py  = cy * tailleCellule + y;
                        float mag = magnitude[py][px];
                        float ang = orientation[py][px];
                        // Le bin est déterminé par l'angle, pondéré par la magnitude
                        int bin = (int)(ang / (180.0f / numBins));
                        if (bin >= numBins) bin = numBins - 1;
                        histCellule[bin] += mag;
                    }
                }
                // Copie de l'histogramme de la cellule dans le vecteur final
                for (float v : histCellule) {
                    vecteurHOG[indexVecteur++] = v;
                }
            }
        }
        return vecteurHOG;
    }

    /**
     * Normalise un vecteur à norme L2 unitaire.
     * Si la norme est quasi-nulle (vecteur nul), retourne le vecteur sans modification
     * pour éviter une division par zéro.
     */
    private static float[] normaliserVecteur(float[] v) {
        double norme = 0.0;
        for (float val : v) norme += val * val;
        norme = Math.sqrt(norme);
        if (norme < 1e-8) return v;
        float[] normalise = new float[v.length];
        for (int i = 0; i < v.length; i++) normalise[i] = (float)(v[i] / norme);
        return normalise;
    }

    /**
     * Charge toutes les images des sous-dossiers de classes (cat/, dog/, wild/)
     * depuis le répertoire de base indiqué, et leur associe le label entier correspondant.
     * Seuls les formats .jpg et .png sont pris en charge.
     */
    static List<Image> chargerImages(String baseDir)
    {
        List<Image> liste   = new ArrayList<>();
        boolean estGris     = typeCouleurActif.equals("GRI");

        for (int c = 0; c < CLASSES.length; c++) {
            String dossier    = baseDir + CLASSES[c] + "/";
            List<String> fichiers = Image.listeFichiers(dossier);
            if (fichiers == null || fichiers.isEmpty()) {
                System.out.printf("  Dossier vide ou introuvable : %s\n", dossier);
                continue;
            }
            int count = 0;
            for (String chemin : fichiers) {
                String low = chemin.toLowerCase();
                if (!low.endsWith(".jpg") && !low.endsWith(".png")) continue;
                liste.add(new Image(chemin, LABELS[c], estGris));
                count++;
            }
            System.out.printf("  Vectorisation | %-5s : %d fichiers chargés\n", CLASSES[c], count);
        }
        return liste;
    }

    /**
     * Instancie le neurone du type sélectionné par l'utilisateur.
     * Retourne un NeuroneSigmoide par défaut si le type est inconnu.
     */
    static Neurone creerNeurone(int nbEntrees)
    {
        return switch (typeNeuroneActif.toLowerCase()) {
            case "heavyside" -> new NeuroneHeavyside(nbEntrees);
            case "relu"      -> new NeuroneReLu(nbEntrees);
            default          -> new NeuroneSigmoide(nbEntrees);
        };
    }

    // =========================================================
    //  ENTRAÎNEMENT
    // =========================================================

    /**
     * Phase d'entraînement : itère sur l'ensemble des images d'entraînement
     * pour mettre à jour les poids de chaque neurone par descente de gradient (Perceptron).
     * S'arrête quand la MSE globale passe sous MSE_LIMITE ou après MAX_ITERATIONS tours.
     * Sauvegarde les poids et la configuration à la fin.
     */
    static void train() throws Exception
    {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" PHASE D'ENTRAÎNEMENT");
        System.out.println("════════════════════════════════════════════════════════");

        List<Image> images = chargerImages(DATASET_TRAIN);
        if (images.isEmpty()) { System.err.println(" Annulation : aucune donnée d'entraînement."); return; }

        if (modeDeepShuffleActif) {
            System.out.println("  Mode ordonnancement : Deep Shuffle (mélange à chaque itération)");
            Collections.shuffle(images);
        } else if (modeShuffleActif) {
            System.out.println("  Mode ordonnancement : Mélange aléatoire (Shuffle)");
            Collections.shuffle(images);
        } else {
            System.out.println("  Mode ordonnancement : Séquentiel linéaire");
        }

        float[][] entrees = chargerEntrees(images);
        int nbEntrees     = entrees[0].length;
        System.out.printf("  Caractéristiques par neurone : %d entrées\n\n", nbEntrees);

        Neurone.fixeCoefApprentissage(ETA);

        // Création de 1 ou 3 neurones selon l'architecture choisie
        int nbNeuronesAEntrainer = modeMonoNeuroneActif ? 1 : CLASSES.length;
        Neurone[] coucheNeurones = new Neurone[nbNeuronesAEntrainer];
        for (int k = 0; k < nbNeuronesAEntrainer; k++) {
            coucheNeurones[k] = creerNeurone(nbEntrees);
        }

        System.out.printf(" Lancement de l'optimisation (%d neurone(s) en compétition)...\n", nbNeuronesAEntrainer);

        long tempsDebutTotal   = System.currentTimeMillis();
        double globaleMse      = 1.0;
        int iter               = 0;

        // Boucle d'entraînement : une itération = un passage complet sur toutes les images
        while (globaleMse > MSE_LIMITE && iter < MAX_ITERATIONS) {
            long tempsDebutIter      = System.nanoTime();
            double sommeErreursCarrees = 0.0;

            // En mode Deep Shuffle, les images et leurs vecteurs précalculés sont réordonnés
            // ensemble à chaque itération pour perturber l'ordre de présentation au réseau
            if (modeDeepShuffleActif) {
                List<Integer> indices = new ArrayList<>();
                for (int i = 0; i < images.size(); i++) indices.add(i);
                Collections.shuffle(indices);
                List<Image> imagesMelangees  = new ArrayList<>(images.size());
                float[][]   entreesMelangees = new float[images.size()][];
                for (int i = 0; i < indices.size(); i++) {
                    imagesMelangees.add(images.get(indices.get(i)));
                    entreesMelangees[i] = entrees[indices.get(i)];
                }
                images  = imagesMelangees;
                entrees = entreesMelangees;
            }

            for (int i = 0; i < images.size(); i++) {
                float[] entree   = entrees[i];
                int     labelReel = images.get(i).label();

                for (int k = 0; k < nbNeuronesAEntrainer; k++) {
                    coucheNeurones[k].metAJour(entree);

                    // Cible : 1.0 si l'image appartient à la classe k, 0.0 sinon (one-vs-all)
                    float cible = (labelReel == LABELS[k]) ? 1.0f : 0.0f;
                    float delta = cible - coucheNeurones[k].sortie(); // erreur de prédiction

                    sommeErreursCarrees += (delta * delta);

                    // Mise à jour des poids selon la règle du perceptron (gradient de la MSE)
                    float[] synapses = coucheNeurones[k].synapses();
                    for (int j = 0; j < entree.length; j++) {
                        synapses[j] += entree[j] * ETA * delta;
                    }
                    // Mise à jour du biais (entrée fictive valant 1)
                    coucheNeurones[k].fixeBiais(coucheNeurones[k].biais() + ETA * delta);
                }
            }

            globaleMse = sommeErreursCarrees / (images.size() * nbNeuronesAEntrainer);
            long tempsFinIter     = System.nanoTime();
            double dureeIterMs    = (tempsFinIter - tempsDebutIter) / 1_000_000.0;

            System.out.printf("  [Itération : %5d ──── MSE : %.6f ──── Durée : %6.2f ms]\n",
                iter, globaleMse, dureeIterMs);
            iter++;
        }

        long tempsFinTotal    = System.currentTimeMillis();
        double dureeTotaleSec = (tempsFinTotal - tempsDebutTotal) / 1000.0;
        System.out.printf("\n Fin du processus. Itérations exécutées : %d | Durée globale : %.2f s.\n",
            iter, dureeTotaleSec);

        // Sauvegarde de la configuration et des poids dans le bon sous-dossier
        sauvegarderConfiguration();
        if (modeMonoNeuroneActif) {
            coucheNeurones[0].sauvegarde(saveDir() + "neurone_binaire_cat.txt");
        } else {
            for (int k = 0; k < CLASSES.length; k++) {
                coucheNeurones[k].sauvegarde(saveDir() + "neurone_" + CLASSES[k] + ".txt");
            }
        }
        System.out.println(" Configuration, poids et biais structurels sauvés dans : " + saveDir());

        System.out.println("\n Performances instantanées mesurées sur le Dataset d'Entraînement :");
        afficherMatriceEtCalculs(images, entrees, coucheNeurones);
    }

    // =========================================================
    //  TEST
    // =========================================================

    /**
     * Phase d'évaluation : charge les poids entraînés et mesure les performances
     * sur le dataset de test (jamais vu pendant l'entraînement).
     * Affiche la matrice de confusion et les précisions par classe.
     */
    static void test() throws Exception
    {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" PHASE D'ÉVALUATION (TEST)");
        System.out.println("════════════════════════════════════════════════════════");

        List<Image> images = chargerImages(DATASET_TEST);
        if (images.isEmpty()) { System.err.println(" Annulation : aucune donnée de test."); return; }

        float[][] entrees = chargerEntrees(images);
        int nbEntrees     = entrees[0].length;

        int nbNeuronesACharger = modeMonoNeuroneActif ? 1 : CLASSES.length;
        Neurone[] neurones     = new Neurone[nbNeuronesACharger];

        try {
            if (modeMonoNeuroneActif) {
                neurones[0] = creerNeurone(nbEntrees);
                neurones[0].chargement(saveDir() + "neurone_binaire_cat.txt");
            } else {
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k] = creerNeurone(nbEntrees);
                    neurones[k].chargement(saveDir() + "neurone_" + CLASSES[k] + ".txt");
                }
            }
        } catch (Exception e) {
            System.err.println("\n[ERREUR CRITIQUE] Impossible de charger les poids depuis " + saveDir());
            System.err.println("Détail de l'erreur : " + e.getMessage());
            System.err.println("-> Assurez-vous d'avoir lancé un ENTRAÎNEMENT complet avec la MÊME configuration (Couleur/Traitement) avant d'évaluer.\n");
            return;
        }

        System.out.println("\n Performances structurelles mesurées sur le Dataset de Test :");
        afficherMatriceEtCalculs(images, entrees, neurones);
        System.out.println("\n Évaluation terminée.");
    }

    /**
     * Calcule et affiche la matrice de confusion ainsi que la précision par classe.
     * En mode binaire (1 neurone) : prédit "cat" si sortie >= 0.5, "non-cat" sinon.
     * En mode multiclasse (3 neurones) : la classe retenue est celle du neurone avec la sortie maximale.
     */
    private static void afficherMatriceEtCalculs(List<Image> images, float[][] entrees, Neurone[] neurones) {
        // colsMatrice = 2 (cat / non-cat) en binaire, = 3 (cat / dog / wild) en multiclasse
        int colsMatrice = modeMonoNeuroneActif ? 2 : CLASSES.length;
        int[][] confusion = new int[CLASSES.length][colsMatrice];
        int nbCorrects    = 0;

        for (int i = 0; i < images.size(); i++) {
            float[] entree   = entrees[i];
            Image   img      = images.get(i);
            int     prediction = 0;

            if (modeMonoNeuroneActif) {
                // Seuil de décision à 0.5 pour sigmoïde/relu, >0 pour heavyside
                neurones[0].metAJour(entree);
                float score = neurones[0].sortie();
                prediction = (score >= 0.5f || (typeNeuroneActif.equalsIgnoreCase("heavyside") && score > 0f)) ? 0 : 1;
            } else {
                // Stratégie winner-takes-all : le neurone avec la sortie la plus haute remporte la classe
                float maxSortie = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k].metAJour(entree);
                    if (neurones[k].sortie() > maxSortie) {
                        maxSortie  = neurones[k].sortie();
                        prediction = k;
                    }
                }
            }

            // Retrouve l'index de la classe réelle dans le tableau LABELS
            int labelReel  = img.label();
            int indexReel  = 0;
            for (int k = 0; k < LABELS.length; k++) {
                if (LABELS[k] == labelReel) { indexReel = k; break; }
            }

            confusion[indexReel][prediction]++;

            // Comptage des prédictions correctes (logique différente selon le mode)
            if (modeMonoNeuroneActif) {
                if ((indexReel == 0 && prediction == 0) || (indexReel != 0 && prediction == 1)) nbCorrects++;
            } else {
                if (prediction == indexReel) nbCorrects++;
            }
        }

        // Construction des bordures de la matrice
        String col     = "──────────";
        String sep     = "├" + (col + "┼").repeat(colsMatrice) + col + "┤";
        String haut    = "┌" + (col + "┬").repeat(colsMatrice) + col + "┐";
        String bas     = "└" + (col + "┴").repeat(colsMatrice) + col + "┘";

        System.out.println(haut);
        System.out.printf("│ %-8s │", "Réel\\Pr");
        if (modeMonoNeuroneActif) {
            System.out.printf(" %-8s │ %-8s │\n", "cat", "non-cat");
        } else {
            for (String c : CLASSES) System.out.printf(" %-8s │", c);
            System.out.println();
        }
        System.out.println(sep);

        for (int i = 0; i < CLASSES.length; i++) {
            System.out.printf("│ %-8s │", CLASSES[i]);
            for (int j = 0; j < colsMatrice; j++) {
                System.out.printf(" %-8d │", confusion[i][j]);
            }
            System.out.println();
        }
        System.out.println(bas);

        // Précision par classe (vrais positifs / total de cette classe)
        System.out.println("\n Précision analytique par classe cible :");
        for (int k = 0; k < CLASSES.length; k++) {
            int total = 0;
            for (int j = 0; j < colsMatrice; j++) total += confusion[k][j];

            double prec;
            if (modeMonoNeuroneActif) {
                int vraisResultats = (k == 0) ? confusion[k][0] : confusion[k][1];
                prec = total > 0 ? 100.0 * vraisResultats / total : 0.0;
                System.out.printf("    %-6s : %3d / %3d réussis (%.2f%%)\n", CLASSES[k], vraisResultats, total, prec);
            } else {
                prec = total > 0 ? 100.0 * confusion[k][k] / total : 0.0;
                System.out.printf("    %-6s : %3d / %3d réussis (%.2f%%)\n", CLASSES[k], confusion[k][k], total, prec);
            }
        }
        System.out.printf("\n Précision globale du modèle : %d / %d (%.2f%%)\n",
            nbCorrects, images.size(), 100.0 * nbCorrects / images.size());
    }

    // =========================================================
    //  SORT (TRI DES IMAGES NON ÉTIQUETÉES)
    // =========================================================

    /**
     * Classe automatiquement les images brutes du dossier to_sort/ en utilisant
     * le modèle entraîné. Les images sont copiées dans les sous-dossiers correspondants
     * (cat/, dog/, wild/ ou cat/, non_cat/ selon le mode).
     * Nettoie les anciens résultats de tri avant de commencer.
     */
    static void sort() throws Exception
    {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" DISTRIB-TRI DE FLUX (SORT)");
        System.out.println("════════════════════════════════════════════════════════");

        List<String> fichiers = Image.listeFichiers(DATASET_SORT);
        if (fichiers == null || fichiers.isEmpty()) {
            System.err.println(" Annulation : aucune donnée brute à trier dans " + DATASET_SORT);
            return;
        }

        // Filtrage sur les formats supportés
        List<String> imagesATrier = new ArrayList<>();
        for (String f : fichiers) {
            String low = f.toLowerCase();
            if (low.endsWith(".jpg") || low.endsWith(".png")) imagesATrier.add(f);
        }

        if (imagesATrier.isEmpty()) {
            System.err.println(" Annulation : Aucun format pris en compte (.jpg, .png) trouvé.");
            return;
        }

        // Nettoyage préalable des dossiers de sortie pour ne pas mélanger avec un tri précédent
        String outDir = sortOutDir();
        System.out.println("  Nettoyage des anciens dossiers de tri dans : " + outDir);
        List<String> dossiersANettoyer = new ArrayList<>(Arrays.asList(CLASSES));
        if (modeMonoNeuroneActif) dossiersANettoyer.add("non_cat");

        for (String cl : dossiersANettoyer) {
            File folder = new File(outDir + cl);
            if (folder.exists() && folder.isDirectory()) {
                File[] anciensFichiers = folder.listFiles();
                if (anciensFichiers != null) {
                    for (File f : anciensFichiers) {
                        if (f.isFile()) f.delete();
                    }
                }
            }
        }

        System.out.printf("  %d images détectées dans 'to_sort/'. Début du tri vers '%s'...\n",
            imagesATrier.size(), outDir);

        // Chargement de toutes les images en mémoire et calcul de leurs vecteurs en une seule passe,
        // ce qui permet au cache de fonctionner correctement entre les objets Image créés ici
        // et ceux éventuellement déjà présents depuis une session train/test précédente.
        boolean estGris = typeCouleurActif.equals("GRI");
        List<Image> imagesChargees = new ArrayList<>();
        for (String cheminFichier : imagesATrier) {
            imagesChargees.add(new Image(cheminFichier, -1, estGris));
        }
        float[][] entreesSort = chargerEntrees(imagesChargees);

        int nbEntrees = entreesSort[0].length;

        int nbNeuronesACharger = modeMonoNeuroneActif ? 1 : CLASSES.length;
        Neurone[] neurones     = new Neurone[nbNeuronesACharger];

        try {
            if (modeMonoNeuroneActif) {
                neurones[0] = creerNeurone(nbEntrees);
                neurones[0].chargement(saveDir() + "neurone_binaire_cat.txt");
            } else {
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k] = creerNeurone(nbEntrees);
                    neurones[k].chargement(saveDir() + "neurone_" + CLASSES[k] + ".txt");
                }
            }
        } catch (Exception e) {
            System.err.println(" Erreur critique : Fichier(s) de poids manquant(s) dans " + saveDir() + ". Ré-entraînez le réseau.");
            return;
        }

        // dispatch[k] = nombre d'images envoyées dans le dossier de la classe k.
        // La taille CLASSES.length + 1 réserve un index supplémentaire pour "non_cat" en mode binaire.
        int[] dispatch = new int[CLASSES.length + (modeMonoNeuroneActif ? 1 : 0)];

        for (int i = 0; i < imagesChargees.size(); i++) {
            Image   img       = imagesChargees.get(i);
            float[] entree    = entreesSort[i];

            Path   source     = Paths.get(img.chemin());
            String nomFichier = source.getFileName().toString();
            Path   cible;

            if (modeMonoNeuroneActif) {
                neurones[0].metAJour(entree);
                float score  = neurones[0].sortie();
                boolean estCat = (score >= 0.5f || (typeNeuroneActif.equalsIgnoreCase("heavyside") && score > 0f));

                if (estCat) {
                    cible = Paths.get(outDir + "cat/" + nomFichier);
                    dispatch[0]++;
                } else {
                    cible = Paths.get(outDir + "non_cat/" + nomFichier);
                    dispatch[CLASSES.length]++;
                }
            } else {
                // Sélection de la classe gagnante par sortie maximale (winner-takes-all)
                int   gagnant   = 0;
                float maxSortie = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < CLASSES.length; k++) {
                    neurones[k].metAJour(entree);
                    if (neurones[k].sortie() > maxSortie) {
                        maxSortie = neurones[k].sortie();
                        gagnant   = k;
                    }
                }
                cible = Paths.get(outDir + CLASSES[gagnant] + "/" + nomFichier);
                dispatch[gagnant]++;
            }

            // Création du dossier de destination si nécessaire, puis copie du fichier
            if (!Files.exists(cible.getParent())) {
                Files.createDirectories(cible.getParent());
            }
            Files.copy(source, cible, StandardCopyOption.REPLACE_EXISTING);
        }

        System.out.println("\n Tri fini avec succès ! Synthèse du flux filtré :");
        if (modeMonoNeuroneActif) {
            System.out.printf("  └─> [cat]     : %d images copiées dans '%s'.\n", dispatch[0], outDir + "cat/");
            System.out.printf("  └─> [non_cat] : %d images copiées dans '%s'.\n", dispatch[CLASSES.length], outDir + "non_cat/");
        } else {
            for (int k = 0; k < CLASSES.length; k++) {
                System.out.printf("  └─> [%-5s]   : %d images copiées dans '%s'.\n",
                    CLASSES[k], dispatch[k], outDir + CLASSES[k] + "/");
            }
        }
    }

    // =========================================================
    //  CLEAR (NETTOYAGE DES SORTIES DE TRI)
    // =========================================================

    /**
     * Supprime les fichiers images précédemment copiés par sort() dans les dossiers
     * naming/binaire/ et/ou naming/multiple/, selon le choix de l'utilisateur.
     * Les dossiers eux-mêmes sont conservés.
     */
    static void clear(Scanner sc)
    {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println(" NETTOYAGE DES SORTIES DE TRI (CLEAR)");
        System.out.println("════════════════════════════════════════════════════════");

        String[] optionsClear = {
            "Binaire uniquement ("  + DATASET_SORT_OUT_BINAIRE  + ")",
            "Multiple uniquement (" + DATASET_SORT_OUT_MULTIPLE + ")",
            "Les deux (Binaire + Multiple)"
        };
        String choixClear = menuChoix(sc, "Que voulez-vous nettoyer ?", optionsClear);

        List<String> dirsANettoyer = new ArrayList<>();
        if (choixClear.startsWith("Binaire"))  dirsANettoyer.add(DATASET_SORT_OUT_BINAIRE);
        if (choixClear.startsWith("Multiple")) dirsANettoyer.add(DATASET_SORT_OUT_MULTIPLE);
        if (choixClear.startsWith("Les deux")) {
            dirsANettoyer.add(DATASET_SORT_OUT_BINAIRE);
            dirsANettoyer.add(DATASET_SORT_OUT_MULTIPLE);
        }

        // Tous les sous-dossiers possibles : les 3 classes + non_cat (mode binaire)
        List<String> sousDossiers = new ArrayList<>(Arrays.asList(CLASSES));
        sousDossiers.add("non_cat");

        int totalSupprime = 0;
        for (String dir : dirsANettoyer) {
            System.out.println("\n  Nettoyage de : " + dir);
            for (String sd : sousDossiers) {
                File dossier = new File(dir + sd);
                if (!dossier.exists() || !dossier.isDirectory()) continue;
                File[] fichiers = dossier.listFiles();
                int count = 0;
                if (fichiers != null) {
                    for (File f : fichiers) {
                        if (f.isFile()) { f.delete(); count++; }
                    }
                }
                totalSupprime += count;
                System.out.printf("    └─> [%-8s] : %d fichier(s) supprimé(s)\n", sd, count);
            }
        }

        System.out.printf("\n Total supprimé : %d fichier(s). Nettoyage terminé.\n", totalSupprime);
    }
}