import { Skeleton } from "@mui/material";
import { MetricGrid } from "./MetricGrid";

export function StatCardSkeletonGrid({ count = 4 }: { count?: number }) {
    const columns = count >= 4 ? 4 : count === 3 ? 3 : 2;
    return (
        <MetricGrid columns={columns}>
            {Array.from({ length: count }).map((_, index) => (
                <Skeleton
                    key={index}
                    variant="rounded"
                    height={120}
                    sx={{ borderRadius: 3, width: "100%" }}
                />
            ))}
        </MetricGrid>
    );
}
