import java.io.FileWriter;
import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;

public abstract class Neurone implements iNeurone
{
	// Coefficient de mise à jour des poids,
	// commun (static) à tous les neurones
	private static float eta = 0.0001f;
	// Accesseur en écriture seule, permettant de modifier
	// eta pour tous les neurones pendant l'exécution
	public static void fixeCoefApprentissage(final float nouvelEta) {eta = nouvelEta;}
	
	// Tolérance immuable (final) et générique (car commune à tous les neurones
	// par le mot-clé static) permettant d'accepter la sortie d'un neurone comme valable
	public static final float ToleranceSortie = 1.e-2f;
	
	// Tableau des poids synaptiques d'un neurone
	private float[] synapses;
	// Biais associé aux poids synaptiques d'un neurone
	private float biais;
	
	// Valeur de sortie d'un neurone (à "Not A Number" par défaut)
	private float etatInterne = Float.NaN;
	
	// Fonction d'activation d'un neurone ; à modifier par héritage,
	// c'est d'ailleurs le but ici du qualificateur abstract, qui dit que cette
	// méthode n'est pas implémentée => à faire dans un ou plusieurs classes filles
	// activation est protected car elle n'a pas à être vue de l'extérieur,
	// mais doit être redéfinie dans les classes filles
	protected abstract float activation(final float valeur);

	// Constructeur d'un neurone
	public Neurone(final int nbEntrees)
	{
		synapses = new float[nbEntrees];
		// On initialise tous les poids de manière alétoire
		for (int i = 0; i < nbEntrees; ++i)
			synapses[i] = (float)(Math.random()*2.-1.);
		// On initialise le biais de manière aléatoire
		biais = (float)(Math.random()*2.-1.);
	}

	// Accesseur pour la valeur de sortie
	public float sortie() {return etatInterne;}
	
	// Donne accès en lecture-écriture aux valeurs des poids synaptiques
	public float[] synapses() {return synapses;}
	// Donne accès en lecture à la valeur du biais
	public float biais() {return biais;}
	// Donne accès en écriture à la valeur du biais
	public void fixeBiais(final float nouveauBiais) {biais = nouveauBiais;}
	
	// Calcule la valeur de sortie en fonction des entrées, des poids synaptiques,
	// du biais et de la fonction d'activation
	public void metAJour(final float[] entrees)
	{
		// On démarre en extrayant le biais
		float somme = biais();
		
		// Puis on ajoute les produits entrée-poids synaptique
		for (int i = 0; i < synapses().length; ++i)
			somme += entrees[i]*synapses()[i];
		
		// On fixe la sortie du neurone relativement à la fonction d'activation
		etatInterne = activation(somme);
	}
	
	// Fonction d'apprentissage relative à la mse
	public void apprentissage(final float[][] entrees, final float[] resultats, final float MSElimite)
	{
		double mse = 0.;
		int iter = 0;
		do
		{
			mse = 0.;
			for (int i = 0; i < entrees.length; ++i)
			{
				final float[] entree = entrees[i];
				metAJour(entree);
				final float delta = resultats[i]-sortie();
				mse += delta * delta;
				for (int j = 0; j < entree.length; ++j)
					synapses()[j] += entree[j]*eta*delta;
				fixeBiais(biais()+eta*delta);
			}
			mse /= entrees.length;
			System.out.printf("Itération %d, mse:  %.6f\n", iter, mse);
			iter += 1;
		}
		while (mse > MSElimite);
	}

	public void sauvegarde(String chemin) // optionel
	{
		try
		{
			FileWriter writer = new FileWriter(chemin);
			for (float x : synapses)
			{
				writer.write(String.valueOf(x) + "\n");
			}
			writer.write(String.valueOf(biais) + "\n");
			writer.close();
			System.out.println("Sauvegarde réussie dans le fichier: " + chemin);
		}
		catch (IOException e)
		{
			System.out.println("Impossible de sauvegarder le neurone dans: " + chemin);
			e.printStackTrace();
		}
	}

	public void chargement(String chemin) // optionel
	{
		try(BufferedReader br = new BufferedReader(new FileReader(chemin)))
		{
			// On remplit chaque poids synaptique avec une valeur par ligne
			for (int i = 0; i < synapses.length; ++i)
			{
				synapses[i] = Float.valueOf(br.readLine());
			}
			// La dernière valeur lue sert de biais
			biais = Float.valueOf(br.readLine());
			System.out.println("Chargement réussi depuis le fichier: " + chemin);
		}
		catch (Exception e)
		{
			System.out.println("Impossible de charger le neurone depuis: " + chemin);
			e.printStackTrace();
		}
	}
}
