import { Box, Skeleton } from "@mui/material";

export function StatCardSkeletonGrid() {
    return (
        <Box
            sx={{
                display: "grid",
                gap: 2,
                gridTemplateColumns: {
                    xs: "1fr",
                    sm: "repeat(2, minmax(0, 1fr))",
                    xl: "repeat(4, minmax(0, 1fr))",
                },
            }}
        >
            {Array.from({ length: 4 }).map((_, index) => (
                <Skeleton
                    key={index}
                    variant="rounded"
                    height={120}
                    sx={{ borderRadius: 3 }}
                />
            ))}
        </Box>
    );
}
