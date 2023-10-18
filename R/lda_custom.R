#' Custom LDA function
#'
#' @param x a matrix
#' @param y a matrix
#' @return a custom LDA function
#' @export
#'
#' @examples
#' # put your awesome examples here
lda_custom <- function(x, y) {
  classes <- unique(y)
  num_classes <- length(classes)
  num_features <- ncol(x)

  # Compute class priors
  class_priors <- table(y) / length(y)

  # Compute class means and covariance matrices
  class_means <- matrix(0, nrow = num_classes, ncol = num_features)
  class_cov <- array(0, dim = c(num_features, num_features, num_classes))

  for (i in 1:num_classes) {
    class_subset <- x[y == classes[i], ]
    class_means[i, ] <- colMeans(class_subset)
    class_cov[,,i] <- cov(class_subset)
  }

  # Compute the overall mean
  overall_mean <- colMeans(x)

  # Compute within-class scatter matrix
  within_class_scatter <- array(0, dim = c(num_features, num_features))
  for (i in 1:num_classes) {
    within_class_scatter <- within_class_scatter + class_cov[,,i] * sum(y == classes[i])
  }

  # Compute between-class scatter matrix
  between_class_scatter <- array(0, dim = c(num_features, num_features))
  for (i in 1:num_classes) {
    between_class_scatter <- between_class_scatter + (class_means[i,] - overall_mean) %*% t(class_means[i,] - overall_mean) * sum(y == classes[i])
  }

  # Solve the eigenvalue problem
  eigenvalues <- eigen(solve(within_class_scatter) %*% between_class_scatter)$values
  eigenvectors <- eigen(solve(within_class_scatter) %*% between_class_scatter)$vectors

  # Sort eigenvalues and corresponding eigenvectors in descending order
  sorted_indices <- order(eigenvalues, decreasing = TRUE)
  eigenvalues <- eigenvalues[sorted_indices]
  eigenvectors <- eigenvectors[, sorted_indices]

  # Compute transformed features
  lda_components <- eigenvectors
  lda_features <- as.matrix(x) %*% lda_components

  # Return results
  result <- list(scaling = lda_components, means = class_means, priors = class_priors)
  class(result) <- "lda"
  return(result)
}
