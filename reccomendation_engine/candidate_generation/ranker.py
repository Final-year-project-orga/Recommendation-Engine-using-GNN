class Ranker:

    def __init__(self,similarity_weight=0.8,rating_weight=0.2):

        self.similarity_weight = similarity_weight
        self.rating_weight = rating_weight

    def rank_candidates(self,candidates):
        """
        candidates = [
            {
                'item_id': ...,
                'name': ...,
                'rating': ...,
                'distance': ...
            }
        ]
        """

        ranked_candidates = []

        for candidate in candidates:

            distance = candidate["distance"]

            # Convert distance to similarity
            similarity_score = 1 / (1 + distance)

            rating_score = (candidate["rating"] / 5.0)

            final_score = (self.similarity_weight* similarity_score + self.rating_weight * rating_score)

            candidate["similarity_score"] = round(similarity_score, 4 )

            candidate["final_score"] = round(final_score,4)

            ranked_candidates.append(candidate)

        ranked_candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True)

        return ranked_candidates

    def get_top_n(self,candidates,n=20):
        """
        Return Top N recommendations
        """

        ranked = self.rank_candidates(candidates )

        return ranked[:n]