require(httr)
require(jsonlite)

###
# Multiple IDs to convert - use a POST request
###
url = "https://biotools.fr/human/ensembl_symbol_converter/"
ids = read.csv("genes.txt", header=FALSE)
ids = as.vector(ids$V1)

ids_json <- toJSON(ids)

body <- list(api=1, ids=ids_json)
r <- POST(url, body = body)

output = fromJSON( content(r, "text"), flatten=TRUE)
output[sapply(output, is.null)] <- NA
output = unlist(output)

write.csv(output, "genes_new.txt", row.names = FALSE)
