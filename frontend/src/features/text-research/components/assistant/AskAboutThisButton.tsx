import { Button } from "@mui/material";

type Props = {
    label?: string;
    question: string;
    intent?: string;
    onAsk: (payload: { question: string; intent?: string }) => void;
    disabled?: boolean;
};

/** Contextual entry point into Ask Corpus (does not mutate scientific results). */
export function AskAboutThisButton({
    label = "Ask about this",
    question,
    intent = "evidence",
    onAsk,
    disabled,
}: Props) {
    return (
        <Button
            size="small"
            variant="outlined"
            disabled={disabled}
            onClick={() => onAsk({ question, intent })}
            aria-label={label}
        >
            {label}
        </Button>
    );
}

export default AskAboutThisButton;
