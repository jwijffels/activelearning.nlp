## active learnng for texts: activelearning.nlp
## 1. global texts: naive bayes, SVM, glmnet, 
##    similarity based on words / content (word2vec / doc2vec / svd_similarity)
## 2. named entity recognition: nametagger + crfsuite

########## QUERY
## most variance in bagging prediction
## most uncertain using kullback-leibner (entropy)
## most certain on 1 side
## most similar semantically based on words
## contains > regular expressions or does not contain
## most uncertain
is_tif <- function(x){
  inherits(x, "data.frame") && all(c("doc_id", "text") %in% colnames(x))
}

activelearning <- function(X, Y, data = X,
                           type = c("grep", "stringdist", "naive_bayes", "glmnet", "svm", "crfsuite", "nametagger", "svd_similarity", "word2vec", "doc2vec"), 
                           FUN = identity, AGG = max,
                           size = nrow(data), pattern, object, ...){
  ##
  ## data should always be a TIF data.frame
  ##
  stopifnot(is_tif(data))
  
  ## currently only with Y as a vector
  type <- match.arg(type)
  
  idx   <- list()
  idx$known   <- which(!is.na(Y))
  idx$unknown <- which(is.na(Y))
  if(length(idx$unknown) > size){
    idx$unknown <- sample(idx$unknown, size = size)
  }
  ldots <- list(...)
  ## type grep: needs doc_id / text
  ## type naive_bayes/glmnet/svm: needs dtm
  ## type word2vec
  if(type == "grep"){
    ##
    ## Similarity based on regular expression
    ##
    stopifnot(is.character(pattern) && length(pattern) == 1)
    data_transformed <- FUN(data)
    ## TIF data.frame with fields doc_id and text
    stopifnot(is_tif(data_transformed))
    stopifnot(nrow(data_transformed) == length(Y))
    data_transformed <- data_transformed[idx$unknown, , drop = FALSE]
    similarity <- grepl(x = data_transformed$text, pattern = pattern, ...)
    similarity <- as.numeric(similarity)
    out   <- list(type = type, pattern = pattern, args = ldots, idx = idx, data = data, similarity = similarity)
  }else if(type == "stringdist"){
    ##
    ## Similarity based on maximum string similarity
    ##
    stopifnot(is.character(pattern) && length(pattern) == 1)
    data_transformed <- FUN(data)
    if(is.data.frame(data_transformed)){
      ## TIF data.frame with fields doc_id and text
      stopifnot(is_tif(data_transformed))
      stopifnot(nrow(data_transformed) == length(Y))
      data_transformed <- data_transformed[idx$unknown, , drop = FALSE]
      similarity <- stringdist::stringsim(a = data_transformed$text, b = pattern, ...)  
    }else{
      ## list of words: take the highest similarity
      data_transformed <- data_transformed[idx$unknown]
      similarity <- sapply(data_transformed, FUN=function(x, pattern, ...){
        similarity <- stringdist::stringsim(a = x, b = pattern, ...)    
        similarity <- AGG(similarity)
        similarity
      }, pattern = pattern, ...)
    }
    out   <- list(type = type, pattern = pattern, args = ldots, idx = idx, data = data, similarity = similarity)
  }else if(type == "naive_bayes"){
    #model <- fastNaiveBayes::fastNaiveBayes(X, Y, ...)
    .NotYetImplemented()
  }else if(type == "glmnet"){
    ##
    ## Similarity based on penalised regression wrt target Y
    ##
    if(missing(X)){
      X <- FUN(data)
    }
    stopifnot(nrow(X) == length(Y))
    model       <- glmnet::cv.glmnet(x = X[idx$known, , drop = FALSE], y = Y[idx$known], ...)
    newdata     <- X[idx$unknown, , drop = FALSE]
    similarity  <- predict(model, newdata, s = "lambda.1se", type = "response")
    similarity  <- drop(similarity)
    out         <- list(type = type, pattern = pattern, args = ldots, idx = idx, data = data, similarity = similarity, model = model)
    # label_order <- apply(scores, MARGIN = 1, FUN = entropy::entropy.plugin)
    # label_order <- data.frame(doc_id = names(label_order),
    #                           entropy = round(as.numeric(label_order), 5),
    #                           stringsAsFactors = FALSE)
    # label_order <- label_order[order(label_order$entropy, decreasing = TRUE), ]
    # head(label_order, 10)
  }else if(type == "svm"){
    .NotYetImplemented()
  }else if(type == "crfsuite"){
    .NotYetImplemented()
  }else if(type == "nametagger"){
    .NotYetImplemented()
  }else if(type == "svd_similarity"){
    ##
    ## Similarity based on singular value decomposition
    ##
    data_transformed <- FUN(data)
    ## should be a sparse matrix
    dtm_svd <- function(dtm, dim = 5, ...){
      SVD <- RSpectra::svds(dtm, nu = 0, k = dim, ...)
      rownames(SVD$v) <- colnames(dtm)
      SVD$v
    }
    embedding  <- dtm_svd(data_transformed, dim = ldots$dim)
    similarity <- udpipe::dtm_svd_similarity(data_transformed, embedding = embedding, weights = pattern)
    similarity <- similarity$similarity$similarity[idx$unknown]
    out        <- list(type = type, pattern = pattern, args = ldots, idx = idx, data = data, similarity = similarity, model = embedding)
  }else if(type == "word2vec"){
    ##
    ## Similarity based on doc2vec to the pattern string
    ##
    data_transformed <- FUN(data)
    ## list of words, get embedding of document and pattern and find similarity
    stopifnot(is.list(data_transformed))
    stopifnot(is.character(pattern) && length(pattern) == 1)
    pattern_vector   <- word2vec::doc2vec(object, newdata = pattern) ## works also on multi-word expressions separated by space
    data_transformed <- data_transformed[idx$unknown]
    similarity       <- sapply(data_transformed, FUN=function(x, pattern_vector){
      wordvectors <- predict(object, newdata = x, type = "embedding")
      similarity  <- word2vec::word2vec_similarity(wordvectors, pattern_vector)
      similarity  <- drop(similarity)
      similarity  <- similarity[!is.na(similarity)]
      if(length(similarity) == 0){
        return(NA_real_)
      }else{
        max(similarity)
      }
    }, pattern_vector = pattern_vector)
    out              <- list(type = type, pattern = pattern, args = ldots, idx = idx, data = data, similarity = similarity, model = object)
  }else if(type == "doc2vec"){
    ##
    ## Similarity based on doc2vec to the pattern string
    ##
    data_transformed <- FUN(data)
    ## list of words, get embedding of document and pattern and find similarity
    stopifnot(is.list(data_transformed))
    stopifnot(is.character(pattern) && length(pattern) == 1)
    pattern_vector   <- word2vec::doc2vec(object, newdata = pattern) ## works also on multi-word expressions separated by space
    data_transformed <- data_transformed[idx$unknown]
    doc_vectors      <- word2vec::doc2vec(object, data_transformed, ...)
    similarity       <- word2vec::word2vec_similarity(doc_vectors, pattern_vector)
    similarity       <- drop(similarity)
    out              <- list(type = type, pattern = pattern, args = ldots, idx = idx, data = data, similarity = similarity, model = object)
  }
  class(out) <- "activelearning"
  out
}
sample <- function(x, ...){
  if(inherits(x, "activelearning")){
    UseMethod("sample")
  }else{
    base::sample(x, ...)
  }
}
sample.activelearning <- function(x, type = c("random", "entropy", "bagging", "committee")){
  type <- match.arg(type)
  ## TODO
  records <- x$data[x$idx$unknown, ]
  records$similarity <- x$similarity
  records <- records[order(records$similarity, decreasing = TRUE), , drop = FALSE]
  records
}

library(word2vec)
library(text2vec)
data("movie_review", package = "text2vec")
DB <- data.frame(doc_id = movie_review$id, 
                 text = movie_review$review, 
                 target = ifelse(rnorm(n = nrow(movie_review)) > 0.1, movie_review$sentiment, NA), 
                 stringsAsFactors = FALSE)
as_dtm <- function(docs){
  docs$token <- tolower(docs$text)
  x <- strsplit.data.frame(docs, "token", "doc_id", split = "[[:space:][:punct:][:digit:]]+")
  x <- document_term_frequencies(x)
  x <- document_term_matrix(x)
  x <- dtm_remove_lowfreq(x, minfreq = 10)
  x <- dtm_conform(x, rows = docs$doc_id)
  x
}
txt_clean_w2v <- function(x){
  text <- x
  text <- gsub("[^[:alnum:]]", " ", text)
  text <- gsub(" +", " ", text)
  text <- tolower(text)
  text <- trimws(text)
  text
}
text <- txt_clean_w2v(DB$text)
w2v  <- word2vec::word2vec(x = text, dim = 10, iter = 20, type = "cbow", min_count = 500)
summary(w2v)
predict(w2v, c("music", "acting"), type = "nearest")


model <- activelearning(data = DB, Y = DB$target, type = "grep", pattern = "bad", ignore.case = TRUE)
model <- activelearning(data = DB, Y = DB$target, type = "stringdist", pattern = "bad", FUN = function(x){
  x <- setNames(tolower(x$text), x$doc_id)
  strsplit(x, split = " ")
})
model <- activelearning(data = DB, Y = DB$target, type = "glmnet", family = "binomial", pattern = "bad", 
                        FUN = function(docs) as_dtm(docs))
model <- activelearning(data = DB, Y = DB$target, type = "svd_similarity", dim = 20,
                        pattern = setNames(c(1, 1, -1, -1), c("good", "fantastic", "bad", "worse")), 
                        FUN = function(docs) as_dtm(docs))
model <- activelearning(data = DB, Y = DB$target, type = "word2vec", object = w2v, pattern = "horrible", 
                        FUN = function(docs){
                          docs <- setNames(txt_clean_w2v(docs$text), docs$doc_id)
                          strsplit(docs, split = " +")
                        }, AGG = function(x) quantile(x, 0.9))
model <- activelearning(data = DB, Y = DB$target, type = "doc2vec", object = w2v, pattern = "bad", 
                        FUN = function(docs){
                          docs <- setNames(txt_clean_w2v(docs$text), docs$doc_id)
                          strsplit(docs, split = " +")
                        })



test <- sample(model)
MASS::truehist(test$similarity)
head(test$text, 1)
tail(test$text, 1)
head(test$text, 2)
tail(test$text, 2)
