export interface EmailRecord {
    id?: number;
    content: string;
    prediction: "spam" | "not_spam";
    confidence?: number;
    created_at?: Date;
  }