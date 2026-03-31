import { memo } from "react";
import { Plus, X, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";

export interface CustomMetadataInputProps {
  metadata: { key: string; value: string }[];
  onChange: (metadata: { key: string; value: string }[]) => void;
}

export const CustomMetadataInput = memo(function CustomMetadataInput({
  metadata,
  onChange,
}: CustomMetadataInputProps) {
  const handleAdd = () => {
    onChange([...metadata, { key: "", value: "" }]);
  };

  const handleRemove = (index: number) => {
    const newMeta = [...metadata];
    newMeta.splice(index, 1);
    onChange(newMeta);
  };

  const handleChange = (index: number, field: "key" | "value", val: string) => {
    const newMeta = [...metadata];
    newMeta[index] = { ...newMeta[index], [field]: val };
    onChange(newMeta);
  };

  const validCount = metadata.filter((m) => m.key.trim() && m.value.trim()).length;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn(
            "w-full h-7 text-[11px] px-2 gap-1.5 border-dashed",
            validCount > 0 ? "border-primary/50 text-primary bg-primary/5" : "text-muted-foreground"
          )}
        >
          <Settings2 className="w-3.5 h-3.5" />
          {validCount > 0 ? `Custom Metadata (${validCount})` : "Add Custom Metadata"}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-64 p-3" align="start">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-xs">Upload Metadata</h4>
            <Button variant="ghost" size="sm" onClick={handleAdd} className="h-6 text-[10px] px-2 h-6">
              <Plus className="w-3 h-3 mr-1" /> Add
            </Button>
          </div>
          
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {metadata.length === 0 ? (
              <p className="text-[10px] text-muted-foreground text-center py-2">
                No custom metadata. These will be added to newly uploaded files.
              </p>
            ) : (
              metadata.map((item, i) => (
                <div key={i} className="flex items-center gap-1.5">
                  <Input
                    placeholder="Key"
                    value={item.key}
                    onChange={(e) => handleChange(i, "key", e.target.value)}
                    className="h-7 text-xs flex-1"
                  />
                  <Input
                    placeholder="Value"
                    value={item.value}
                    onChange={(e) => handleChange(i, "value", e.target.value)}
                    className="h-7 text-xs flex-1"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleRemove(i)}
                    className="h-7 w-7 text-muted-foreground hover:text-destructive shrink-0"
                  >
                    <X className="w-3.5 h-3.5" />
                  </Button>
                </div>
              ))
            )}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
});
