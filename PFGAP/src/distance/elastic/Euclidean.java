package distance.elastic;

import java.io.Serial;
import java.io.Serializable;

/**
 * Euclidean distance implementations for primitive and boxed numeric vectors.
 *
 * <p>The primitive overloads used by proximity-tree routing return squared
 * Euclidean distance. This preserves the existing behavior and avoids an
 * unnecessary square-root operation when only relative distance ordering is
 * required.</p>
 *
 * <p>The selected-dimension primitive overload interprets positions in a
 * {@code double[]} instance as tabular features. It evaluates only the
 * supplied feature indices and does not allocate a reduced copy of either
 * input vector.</p>
 *
 * <p>The boxed overload used by existing KNN-imputation workflows returns
 * ordinary Euclidean distance.</p>
 *
 * <p>Methods remain synchronized pending a project-wide audit of distance
 * instance ownership and concurrency behavior.</p>
 */
public class Euclidean implements Serializable {

	@Serial
	private static final long serialVersionUID =
			1L;

	public Euclidean() {
	}

	/**
	 * Computes squared Euclidean distance over every primitive vector
	 * component.
	 *
	 * <p>The calculation may stop after the accumulated squared distance
	 * exceeds {@code bestSoFar}. The returned value is therefore sufficient
	 * for nearest-exemplar comparison, but it may be a partially accumulated
	 * value when early abandoning occurs.</p>
	 *
	 * @param first primitive first vector
	 * @param second primitive second vector
	 * @param bestSoFar current best squared distance
	 * @return squared Euclidean distance, possibly early-abandoned
	 */
	public synchronized double distance(
			Object first,
			Object second,
			double bestSoFar
	) {
		double[] firstValues =
				(double[]) first;

		double[] secondValues =
				(double[]) second;

		double total =
				0.0;

		for (int index = 0;
			 index < firstValues.length
					 && total <= bestSoFar;
			 index++) {

			double difference =
					firstValues[index]
							- secondValues[index];

			total +=
					difference * difference;
		}

		return total;
	}

	/**
	 * Computes squared Euclidean distance over selected primitive vector
	 * components.
	 *
	 * <p>No subarray, mask, or reduced vector is allocated. The supplied
	 * selected-dimension array is expected to contain valid, distinct indices
	 * and is treated as read-only.</p>
	 *
	 * @param first primitive first vector
	 * @param second primitive second vector
	 * @param bestSoFar current best squared distance
	 * @param selectedDimensions selected tabular-feature indices, or null to
	 *                           use every feature
	 * @return selected-feature squared Euclidean distance, possibly
	 *         early-abandoned
	 */
	public synchronized double distance(
			Object first,
			Object second,
			double bestSoFar,
			int[] selectedDimensions
	) {
		if (selectedDimensions == null) {
			return distance(
					first,
					second,
					bestSoFar
			);
		}

		double[] firstValues =
				(double[]) first;

		double[] secondValues =
				(double[]) second;

		double total =
				0.0;

		for (int selectedPosition = 0;
			 selectedPosition < selectedDimensions.length
					 && total <= bestSoFar;
			 selectedPosition++) {

			int feature =
					selectedDimensions[selectedPosition];

			double difference =
					firstValues[feature]
							- secondValues[feature];

			total +=
					difference * difference;
		}

		return total;
	}

	/**
	 * Computes ordinary Euclidean distance over every boxed vector component.
	 *
	 * <p>This overload retains the existing behavior used by KNN-imputation
	 * workflows.</p>
	 *
	 * @param first boxed first vector
	 * @param second boxed second vector
	 * @return ordinary Euclidean distance
	 */
	public synchronized double distance(
			Object first,
			Object second
	) {
		Double[] firstValues =
				(Double[]) first;

		Double[] secondValues =
				(Double[]) second;

		double total =
				0.0;

		for (int index = 0;
			 index < firstValues.length;
			 index++) {

			double difference =
					firstValues[index]
							- secondValues[index];

			total +=
					difference * difference;
		}

		return Math.sqrt(
				total
		);
	}

	/**
	 * Computes ordinary Euclidean distance over selected boxed vector
	 * components.
	 *
	 * <p>This overload is provided for selected-feature boxed-data workflows.
	 * It preserves the ordinary, rather than squared, Euclidean result of the
	 * existing boxed overload.</p>
	 *
	 * @param first boxed first vector
	 * @param second boxed second vector
	 * @param selectedDimensions selected tabular-feature indices, or null to
	 *                           use every feature
	 * @return selected-feature ordinary Euclidean distance
	 */
	public synchronized double distance(
			Object first,
			Object second,
			int[] selectedDimensions
	) {
		if (selectedDimensions == null) {
			return distance(
					first,
					second
			);
		}

		Double[] firstValues =
				(Double[]) first;

		Double[] secondValues =
				(Double[]) second;

		double total =
				0.0;

		for (int selectedPosition = 0;
			 selectedPosition < selectedDimensions.length;
			 selectedPosition++) {

			int feature =
					selectedDimensions[selectedPosition];

			double difference =
					firstValues[feature]
							- secondValues[feature];

			total +=
					difference * difference;
		}

		return Math.sqrt(
				total
		);
	}
}